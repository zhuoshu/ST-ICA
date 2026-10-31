import torch
import torch.nn as nn

class FRPlusModule(nn.Module):
    """幅相多组校正模块"""
    def __init__(self, num_nodes, freq_bins, groups=4):
        super().__init__()
        print('check:',freq_bins, groups)
        self.groups = groups
        # 按频点分组，每组大小约 freq_bins//groups
        self.group_size = freq_bins // groups
        # 幅值和相位校正参数: [groups, num_nodes, 1]
        self.lambda_amp = nn.Parameter(torch.zeros(groups, num_nodes, 1))
        self.lambda_phi = nn.Parameter(torch.zeros(groups, num_nodes, 1))

    def forward(self, y_pred):
        # y_pred: [B,T,N]
        B, T, N = y_pred.shape
        # y = y_pred[:,0]  # [B,N,T]
        y = torch.permute(y_pred, (0,2,1)) # B N T
        # FFT -> [B,N,M]
        Yf = torch.fft.rfft(y, dim=-1)
        A = torch.abs(Yf)       # 幅值
        P = torch.angle(Yf)     # 相位

        # 构造校正后的复谱
        Yf_corr = torch.zeros_like(Yf)
        for g in range(self.groups):
            start = g * self.group_size
            end = T//2+1 if g==self.groups-1 else (g+1)*self.group_size
            lam_a = self.lambda_amp[g].unsqueeze(0)  # -> [1,N,1]
            lam_p = self.lambda_phi[g].unsqueeze(0)
            # 幅相校正
            A_g = A[:,:,start:end] * (1 + lam_a)
            P_g = P[:,:,start:end] + lam_p
            # 重构该组
            Yf_corr[:,:,start:end] = A_g * torch.exp(1j * P_g)

        # iFFT 恢复时域 [B,N,T]
        y_time = torch.fft.irfft(Yf_corr, n=T, dim=-1)
        # return y_time.unsqueeze(1)  # [B,1,N,T]
        return y_time.permute((0,2,1)) # B T N



class NodeFRParams(nn.Module):
    """某一个 node_id 对应的幅相校正参数"""
    def __init__(self, groups: int):
        super().__init__()
        # [groups, 1]
        self.lambda_amp = nn.Parameter(torch.zeros(groups, 1))
        self.lambda_phi = nn.Parameter(torch.zeros(groups, 1))

class DynamicFRPlusModule(nn.Module):
    """幅相多组校正模块, 节点参数按 node_id 动态增量"""
    def __init__(self, freq_bins, groups=4):
        super().__init__()
        self.groups = groups
        self.freq_bins = freq_bins
        # 仍保留原来的分组逻辑, 但 forward 里会用实际 M 做兜底
        self.group_size = max(1, freq_bins // groups)

        # key: str(node_id), value: NodeFRParams
        self.node_params = nn.ModuleDict()

    @torch.no_grad()
    def ensure_nodes(self, node_ids, device=None):
        """确保这些 node_id 都有对应子模块, 新增的会创建并放到 device 上"""
        new_keys = []
        for nid in node_ids:
            k = str(int(nid) if torch.is_tensor(nid) else nid)
            if k not in self.node_params:
                self.node_params[k] = NodeFRParams(self.groups)
                new_keys.append(k)

        if device is not None:
            for k in new_keys:
                self.node_params[k].to(device)
        return new_keys

    def _gather_node_lambdas(self, node_ids, device):
        """
        返回:
          lam_a: [groups, N, 1]
          lam_p: [groups, N, 1]
        """
        lam_a_list, lam_p_list = [], []
        for nid in node_ids:
            k = str(int(nid) if torch.is_tensor(nid) else nid)
            p = self.node_params[k]
            lam_a_list.append(p.lambda_amp)  # [groups,1]
            lam_p_list.append(p.lambda_phi)

        # stack over N
        lam_a = torch.stack(lam_a_list, dim=1).to(device)  # [groups,N,1]
        lam_p = torch.stack(lam_p_list, dim=1).to(device)  # [groups,N,1]
        return lam_a, lam_p

    def forward(self, y_pred, node_ids):
        """
        y_pred: [B,T,N_i, 1]
        node_ids: length N_i, 顺序与 y_pred 的 N 维一致
        """
        B, T, N, C = y_pred.shape
        device = y_pred.device

        # [B,N,T]
        y = y_pred.squeeze(-1).permute(0, 2, 1)

        # FFT -> [B,N,M], M = T//2+1
        Yf = torch.fft.rfft(y, dim=-1)
        A = torch.abs(Yf)
        P = torch.angle(Yf)
        M = Yf.shape[-1]

        # 取出本 stage 节点对应的参数
        lam_a, lam_p = self._gather_node_lambdas(node_ids, device)  # [groups,N,1]

        # 构造校正后的复谱
        Yf_corr = torch.zeros_like(Yf)

        # 兜底, 用实际 M 来定义每组范围, 避免 freq_bins 和 M 不一致导致越界
        group_size = max(1, min(self.group_size, M))

        for g in range(self.groups):
            start = g * group_size
            if start >= M:
                break
            end = M if g == self.groups - 1 else min((g + 1) * group_size, M)

            # -> [1,N,1]
            lam_a_g = lam_a[g].unsqueeze(0)
            lam_p_g = lam_p[g].unsqueeze(0)

            # 幅相校正
            A_g = A[:, :, start:end] * (1.0 + lam_a_g)
            P_g = P[:, :, start:end] + lam_p_g

            Yf_corr[:, :, start:end] = A_g * torch.exp(1j * P_g)

        # iFFT -> [B,N,T] -> [B,T,N]
        y_time = torch.fft.irfft(Yf_corr, n=T, dim=-1)
        return y_time.permute(0, 2, 1).unsqueeze(-1)
