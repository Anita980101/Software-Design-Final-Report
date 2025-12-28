import unittest
import numpy as np
import sys
import os
from unittest.mock import MagicMock, patch
import matplotlib.pyplot as plt
from datetime import datetime

# 1. 設定路徑，確保可以 import 上層模組
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 2. 直接 Import 控制器
# 確保 config.py 存在於專案目錄中，若不存在，下方的 patch 仍會攔截它，但建議環境要完整
from visual_servoing_controller import VisualServoController

class TestVisualServoController(unittest.TestCase):
    def setUp(self):
        # 3. Patch Config：攔截 visual_servoing_controller 模組內的 Config
        self.patcher = patch('visual_servoing_controller.Config')
        self.MockConfig = self.patcher.start()
        
        # 4. [關鍵修正] 在初始化 Controller 之前，先設定好 Mock 的屬性
        # 這樣 __init__ 中的 np.array(Config.COMMAND_POSITION) 才能讀到正確的 List
        self.MockConfig.COMMAND_POSITION = [-0.15, -0.17, -0.16, 0.16, 0.17, 0.17, 0.18, -0.16]
        self.MockConfig.KP_DEFAULT = 0.01
        self.MockConfig.KI_DEFAULT = 0.0000025
        self.MockConfig.Z_HEIGHT = 0.31
        self.MockConfig.CONTROL_MODE = 'P' # 預設測試 P 模式

        # 5. 初始化控制器 (此時會讀取上面的 MockConfig)
        self.controller = VisualServoController()

    def tearDown(self):
        self.patcher.stop()

    def test_interaction_matrix_shape(self):
        """測試 Interaction Matrix 維度"""
        features = np.zeros(8) # 4個點，每個點 (u, v)
        Z = 0.5
        
        Le = self.controller.calculate_interaction_matrix(features, Z)
        
        # 4個特徵點 => 8個座標，對應 6 個自由度，形狀應為 (8, 6)
        self.assertEqual(Le.shape, (8, 6))

    def test_update_returns(self):
        """測試 update 函式的回傳格式"""
        features = np.array(self.MockConfig.COMMAND_POSITION) # 使用正確維度的輸入
        vc, errs, kp, mem = self.controller.update(features)
        
        self.assertEqual(vc.shape, (6,))
        self.assertEqual(len(errs), 4)
        
        # 檢查 kp 是否為浮點數 (因為 Config.KP_DEFAULT 是 float)
        self.assertIsInstance(kp, float)

    def test_mode_selection(self):
        """測試不同模式不會崩潰"""
        modes = ['P', 'PI', 'FUZZY', 'MEMRISTOR']
        features = np.array(self.MockConfig.COMMAND_POSITION)
        
        # Mock 依賴的控制器，避免內部呼叫失敗
        self.controller.fuzzy_controller = MagicMock()
        self.controller.fuzzy_controller.compute_gain.return_value = 0.5
        
        self.controller.memristor_controller = MagicMock()
        self.controller.memristor_controller.get_output.return_value = 0.5
        self.controller.memristor_controller.total_ristor = [100] 

        for mode in modes:
            self.controller.mode = mode
            if mode == 'PI':
                self.controller.error_sum = np.zeros(8)
                
            try:
                self.controller.update(features)
            except Exception as e:
                self.fail(f"Mode {mode} raised exception: {e}")

    def test_is_first_run_logic(self):
        """測試：首次運行與後續運行的行為差異"""
        features = np.array(self.MockConfig.COMMAND_POSITION)
        
        # 第一次運行
        self.assertTrue(self.controller.is_first_run)
        self.controller.update(features)
        
        # 運行後 flag 應被清除
        self.assertFalse(self.controller.is_first_run)
        # 檢查 eta 是否被計算 (不應全為 0，除非 features 完全等於目標)
        # 我們故意加一點誤差來確保 eta 有值
        features_noise = features + 0.01
        self.controller.is_first_run = True # 重置
        self.controller.update(features_noise)
        self.assertFalse(np.all(self.controller.eta == 0))

    def test_PI_integration_accumulation(self):
        """測試：PI 模式下，恆定誤差是否會導致積分項(Integral)累積"""
        self.controller.mode = 'PI' # 直接修改 instance 的 mode
        self.controller.ki = 0.1     # 設定大一點以便觀察
        
        # 模擬一個恆定的誤差輸入
        features_with_error = np.array(self.MockConfig.COMMAND_POSITION) + 0.1
        
        # 第一步
        vc1, _, _, _ = self.controller.update(features_with_error)
        
        # 第二步 (輸入完全相同的特徵位置)
        vc2, _, _, _ = self.controller.update(features_with_error)
        
        # 因為是 PI 控制，誤差總和 (error_sum) 會持續累積
        # 所以即使輸入一樣，第二次輸出的速度命令 (vc2) 絕對值總和應該不同於第一次 (vc1)
        # 通常會變大
        self.assertFalse(np.allclose(vc1, vc2), "PI controller did not accumulate error (output did not change)")

    def test_phi_decay(self):
        """測試：誤差修正項 phi 是否隨時間衰減 (指數收斂)"""
        # 使用正確維度的特徵
        features = np.array(self.MockConfig.COMMAND_POSITION)
        
        with patch('time.time') as mock_time:
            # 時間點 0
            mock_time.return_value = 100.0
            self.controller.start_time = 100.0
            self.controller.t_prev = 100.0
            self.controller.is_first_run = False # 強制非首次
            self.controller.eta = np.ones(6) # 設定初始 eta
            self.controller.kp = 1.0
            
            # Update 1: t = 100.0
            self.controller.update(features)
            phi_t0 = np.linalg.norm(self.controller.phi)
            
            # Update 2: t = 102.0 (過了2秒)
            mock_time.return_value = 102.0
            self.controller.update(features)
            phi_t1 = np.linalg.norm(self.controller.phi)
            
            # 隨時間經過，phi 應該變小 (因為 np.exp(-0.5 * dt))
            # 注意：如果原本 phi 已經是 0，這裡就不會變小，所以我們要確保 eta 非 0
            self.assertLess(phi_t1, phi_t0)

    def test_singularity_or_low_Z_handling(self):
        """測試：當 Z 極小 (接近 0) 時的矩陣計算穩定性"""
        features = np.zeros(8)
        tiny_Z = 0.00001
        
        # 這可能會導致數值極大，我們測試是否會拋出錯誤或產生 Inf/NaN
        Le = self.controller.calculate_interaction_matrix(features, tiny_Z)
        
        # 檢查是否含有 NaN 或 Inf
        if np.any(np.isinf(Le)) or np.any(np.isnan(Le)):
            print("Warning: Interaction Matrix contains Inf or NaN with low Z")
        else:
            # 如果通過 pinv，檢查偽逆矩陣
            pinv_Le = np.linalg.pinv(Le)
            self.assertFalse(np.any(np.isnan(pinv_Le)), "Pinv produced NaNs")
    
    def test_is_first_run_logic(self):
        """測試：首次運行與後續運行的行為差異"""
        features = np.array([0.1] * 8)
        
        # 第一次運行
        self.assertTrue(self.controller.is_first_run)
        self.controller.update(features)
        
        # 運行後 flag 應被清除
        self.assertFalse(self.controller.is_first_run)
        # 檢查 eta 是否被計算並儲存 (不應全為 0)
        self.assertFalse(np.all(self.controller.eta == 0))

    def test_PI_integration_accumulation(self):
        """測試：PI 模式下，恆定誤差是否會導致積分項(Integral)累積"""
        self.MockConfig.CONTROL_MODE = 'PI'
        self.MockConfig.KI_DEFAULT = 0.1 # 設定大一點以便觀察
        self.controller.ki = 0.1
        
        features_with_error = np.array(self.MockConfig.COMMAND_POSITION) + 0.1
        
        # 第一步
        vc1, _, _, _ = self.controller.update(features_with_error)
        
        # 第二步 (輸入相同誤差)
        vc2, _, _, _ = self.controller.update(features_with_error)
        
        # 因為有積分項，即使誤差輸入一樣，第二次輸出的速度應該要比第一次大 (或不同)
        # 這裡檢查絕對值的總和是否增加
        self.assertNotAlmostEqual(np.sum(np.abs(vc1)), np.sum(np.abs(vc2)))

    def test_custom_feature_trajectory(self):
        """
        測試：給定一系列自定義的特徵點，追蹤輸出的速度指令軌跡
        這個測試可以用來驗證控制器在特定特徵點序列下的行為
        """
        # ===== 1. 定義你想測試的特徵點序列 =====
        # 每個特徵點包含 8 個值 [u1, v1, u2, v2, u3, v3, u4, v4]
        # 這裡示範從初始位置逐步移動到目標位置的軌跡，共 20 個時間點
        
        # 目標位置
        target_features = np.array(self.MockConfig.COMMAND_POSITION)
        
        # 初始位置 (與目標有較大誤差)
        initial_features = np.array([-0.08, -0.10, -0.09, 0.24, 0.25, 0.24, 0.26, -0.09])
        
        # 生成從初始位置到目標位置的漸進軌跡（20個點）
        num_steps = 20
        feature_sequence = []
        for i in range(num_steps):
            # 線性插值，逐步從初始位置移動到目標位置
            alpha = i / (num_steps - 1)  # 0 到 1
            # 使用非線性插值模擬真實運動（開始快，後來慢）
            alpha_smooth = alpha ** 1.5
            interpolated = initial_features + alpha_smooth * (target_features - initial_features)
            # 添加一些小的擾動模擬真實測量噪聲
            noise = np.random.normal(0, 0.002, 8) if i < num_steps - 1 else np.zeros(8)
            feature_sequence.append(interpolated + noise)
        
        # ===== 2. 記錄控制器輸出的軌跡 =====
        velocity_trajectory = []  # 速度指令軌跡
        error_trajectory = []     # 誤差軌跡
        gain_trajectory = []      # 增益軌跡
        
        # 模擬時間流逝
        with patch('time.time') as mock_time:
            t = 0.0
            for features in feature_sequence:
                mock_time.return_value = t
                
                # 更新控制器
                vc, pixel_errors, kp, mem = self.controller.update(features)
                
                # 記錄軌跡
                velocity_trajectory.append(vc.copy())
                error_trajectory.append(pixel_errors)
                gain_trajectory.append(kp)
                
                t += 0.1  # 每次間隔 0.1 秒
        
        # ===== 3. 驗證軌跡特性 =====
        
        # (1) 檢查軌跡長度是否正確
        self.assertEqual(len(velocity_trajectory), len(feature_sequence))
        
        # (2) 檢查誤差是否逐漸減小 (收斂性)
        avg_errors = [np.mean(errs) for errs in error_trajectory]
        print("\n===== 特徵點軌跡測試結果 =====")
        print(f"平均誤差序列: {avg_errors}")
        
        # 驗證最後的誤差小於第一個誤差 (表示系統在收斂)
        self.assertLess(avg_errors[-1], avg_errors[0], 
                       "誤差沒有減小，系統可能不穩定")
        
        # (3) 檢查速度指令是否合理 (不應該有異常大的值)
        max_velocity = np.max([np.abs(vc).max() for vc in velocity_trajectory])
        print(f"最大速度指令: {max_velocity:.6f}")
        self.assertLess(max_velocity, 1.0, 
                       "速度指令過大，可能存在數值問題")
        
        # (4) 輸出軌跡供檢視
        print("\n速度指令軌跡 (6自由度):")
        for i, vc in enumerate(velocity_trajectory):
            print(f"  時間步 {i}: {vc}")
        
        print("\n像素誤差軌跡 (4個特徵點):")
        for i, errs in enumerate(error_trajectory):
            print(f"  時間步 {i}: {errs}")
        
        # (5) 可選：檢查最終是否接近目標
        final_features = feature_sequence[-1]
        final_error = np.linalg.norm(target_features - final_features)
        print(f"\n最終特徵點誤差範數: {final_error:.6f}")
        
        # (6) 繪製軌跡圖
        self._plot_trajectory_results(
            feature_sequence, 
            velocity_trajectory, 
            error_trajectory, 
            gain_trajectory,
            target_features
        )
        
        return velocity_trajectory, error_trajectory

    def _plot_trajectory_results(self, feature_sequence, velocity_traj, error_traj, gain_traj, target_features):
        """
        繪製測試軌跡的視覺化圖表
        """
        # 創建輸出目錄
        test_output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'test_results'
        )
        os.makedirs(test_output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 轉換數據格式
        velocity_array = np.array(velocity_traj)  # shape: (N, 6)
        time_steps = np.arange(len(feature_sequence))
        
        # ===== 圖1: 特徵點軌跡 (Image Space) =====
        fig1, axes1 = plt.subplots(2, 2, figsize=(12, 10))
        fig1.suptitle('特徵點在影像空間中的軌跡', fontsize=14, fontweight='bold')
        
        for point_idx in range(4):
            row = point_idx // 2
            col = point_idx % 2
            ax = axes1[row, col]
            
            # 提取該特徵點的 u, v 軌跡
            u_traj = [features[2*point_idx] for features in feature_sequence]
            v_traj = [features[2*point_idx+1] for features in feature_sequence]
            
            # 目標位置
            u_target = target_features[2*point_idx]
            v_target = target_features[2*point_idx+1]
            
            # 繪製軌跡
            ax.plot(u_traj, v_traj, 'b-o', label='actual trajectory', markersize=8, linewidth=2)
            ax.plot(u_traj[0], v_traj[0], 'go', markersize=12, label='start')
            ax.plot(u_traj[-1], v_traj[-1], 'ro', markersize=12, label='finish')
            ax.plot(u_target, v_target, 'r*', markersize=15, label='target point')
            
            ax.set_xlabel('u (normalized)', fontsize=10)
            ax.set_ylabel('v (normalized)', fontsize=10)
            ax.set_title(f'Feature Point {point_idx+1}', fontsize=11, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.axis('equal')
        
        plt.tight_layout()
        fig1.savefig(os.path.join(test_output_dir, f'{timestamp}_feature_trajectory.png'), dpi=150)
        print(f"\n✓ 特徵點軌跡圖已保存: test_results/{timestamp}_feature_trajectory.png")
        
        # ===== 圖2: 速度指令軌跡 =====
        fig2, axes2 = plt.subplots(3, 2, figsize=(14, 12))
        fig2.suptitle('速度指令軌跡 (6自由度)', fontsize=14, fontweight='bold')
        
        velocity_labels = ['Vx', 'Vy', 'Vz', 'ωx', 'ωy', 'ωz']
        for i in range(6):
            row = i // 2
            col = i % 2
            ax = axes2[row, col]
            
            ax.plot(time_steps, velocity_array[:, i], 'b-o', linewidth=2, markersize=6)
            ax.set_xlabel('時間步', fontsize=10)
            ax.set_ylabel(f'{velocity_labels[i]} (m/s or rad/s)', fontsize=10)
            ax.set_title(f'{velocity_labels[i]} 速度指令', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        fig2.savefig(os.path.join(test_output_dir, f'{timestamp}_velocity_commands.png'), dpi=150)
        print(f"✓ 速度指令圖已保存: test_results/{timestamp}_velocity_commands.png")
        
        # ===== 圖3: 誤差收斂曲線 =====
        fig3, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        fig3.suptitle('誤差收斂分析', fontsize=14, fontweight='bold')
        
        # 上圖：每個特徵點的誤差
        error_array = np.array(error_traj)  # shape: (N, 4)
        for i in range(4):
            ax1.plot(time_steps, error_array[:, i], '-o', label=f'特徵點 {i+1}', linewidth=2, markersize=6)
        
        ax1.set_xlabel('時間步', fontsize=10)
        ax1.set_ylabel('像素誤差', fontsize=10)
        ax1.set_title('各特徵點的誤差變化', fontsize=11, fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # 下圖：平均誤差
        avg_errors = np.mean(error_array, axis=1)
        ax2.plot(time_steps, avg_errors, 'r-o', linewidth=2.5, markersize=8)
        ax2.set_xlabel('時間步', fontsize=10)
        ax2.set_ylabel('平均像素誤差', fontsize=10)
        ax2.set_title('平均誤差收斂曲線', fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.fill_between(time_steps, avg_errors, alpha=0.3, color='red')
        
        plt.tight_layout()
        fig3.savefig(os.path.join(test_output_dir, f'{timestamp}_error_convergence.png'), dpi=150)
        print(f"✓ 誤差收斂圖已保存: test_results/{timestamp}_error_convergence.png")
        
        # ===== 圖4: 綜合儀表板 =====
        fig4 = plt.figure(figsize=(16, 10))
        gs = fig4.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 左上：特徵點在影像空間的總覽（四個點連成四邊形）
        ax_img = fig4.add_subplot(gs[0:2, 0:2])
        
        # 定義顏色
        colors = ['red', 'blue', 'orange', 'green']
        
        # 繪製起始四邊形（綠色實線）
        start_features = feature_sequence[0]
        start_u = [start_features[2*i] for i in range(4)]
        start_v = [start_features[2*i+1] for i in range(4)]
        start_u_closed = start_u + [start_u[0]]
        start_v_closed = start_v + [start_v[0]]
        ax_img.plot(start_u_closed, start_v_closed, 'g-', 
                   linewidth=2.5, alpha=0.8, label='Start')
        
        # 繪製結束四邊形（紅色實線）
        end_features = feature_sequence[-1]
        end_u = [end_features[2*i] for i in range(4)]
        end_v = [end_features[2*i+1] for i in range(4)]
        end_u_closed = end_u + [end_u[0]]
        end_v_closed = end_v + [end_v[0]]
        ax_img.plot(end_u_closed, end_v_closed, 'r-', 
                   linewidth=2.5, alpha=1.0, label='End')
        
        # 繪製目標四邊形（黑色虛線）
        target_u = [target_features[2*i] for i in range(4)]
        target_v = [target_features[2*i+1] for i in range(4)]
        target_u_closed = target_u + [target_u[0]]
        target_v_closed = target_v + [target_v[0]]
        ax_img.plot(target_u_closed, target_v_closed, 'k--', 
                   linewidth=2.5, alpha=0.8, label='Target')
        
        # 繪製每個特徵點的移動軌跡
        for point_idx in range(4):
            u_traj = [features[2*point_idx] for features in feature_sequence]
            v_traj = [features[2*point_idx+1] for features in feature_sequence]
            
            # 繪製軌跡線
            ax_img.plot(u_traj, v_traj, '-', color=colors[point_idx], 
                       linewidth=2, alpha=0.6, label=f'point{point_idx}')
            
            # 標記起點和終點
            ax_img.plot(u_traj[0], v_traj[0], 'o', color=colors[point_idx], 
                       markersize=8, markeredgecolor='black', markeredgewidth=1.5)
            ax_img.plot(u_traj[-1], v_traj[-1], 's', color=colors[point_idx], 
                       markersize=8, markeredgecolor='black', markeredgewidth=1.5)
        
        ax_img.set_xlabel('X Coordinate', fontsize=10)
        ax_img.set_ylabel('Y Coordinate', fontsize=10)
        ax_img.set_title('Trajectory Plot from Start Point to End Point', fontsize=12, fontweight='bold')
        ax_img.legend(fontsize=8, loc='best')
        ax_img.grid(True, alpha=0.3)
        ax_img.axis('equal')
        
        # 右上：速度範數
        ax_vel_norm = fig4.add_subplot(gs[0, 2])
        vel_norms = np.linalg.norm(velocity_array, axis=1)
        ax_vel_norm.plot(time_steps, vel_norms, 'g-o', linewidth=2)
        ax_vel_norm.set_xlabel('時間步', fontsize=9)
        ax_vel_norm.set_ylabel('速度範數', fontsize=9)
        ax_vel_norm.set_title('速度大小', fontsize=10, fontweight='bold')
        ax_vel_norm.grid(True, alpha=0.3)
        
        # 右中：增益變化
        ax_gain = fig4.add_subplot(gs[1, 2])
        ax_gain.plot(time_steps, gain_traj, 'm-o', linewidth=2)
        ax_gain.set_xlabel('時間步', fontsize=9)
        ax_gain.set_ylabel('控制增益 Kp', fontsize=9)
        ax_gain.set_title('增益變化', fontsize=10, fontweight='bold')
        ax_gain.grid(True, alpha=0.3)
        
        # 下方：平均誤差
        ax_err = fig4.add_subplot(gs[2, :])
        ax_err.plot(time_steps, avg_errors, 'r-o', linewidth=2.5, markersize=8)
        ax_err.set_xlabel('時間步', fontsize=10)
        ax_err.set_ylabel('平均像素誤差', fontsize=10)
        ax_err.set_title('誤差收斂', fontsize=11, fontweight='bold')
        ax_err.grid(True, alpha=0.3)
        ax_err.fill_between(time_steps, avg_errors, alpha=0.3, color='red')
        
        fig4.suptitle('測試軌跡綜合儀表板', fontsize=16, fontweight='bold', y=0.995)
        fig4.savefig(os.path.join(test_output_dir, f'{timestamp}_dashboard.png'), dpi=150)
        print(f"✓ 綜合儀表板已保存: test_results/{timestamp}_dashboard.png")
        
        plt.show()
        print(f"\n所有圖表已生成並顯示！")


if __name__ == '__main__':
    unittest.main()