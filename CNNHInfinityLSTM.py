import torch.nn.functional as F

def solve_riccati(A, B, Q, R, max_iter=50, eps=1e-6):
    #Differentiable iterative Riccati equation solver (discrete-time)
    P = Q
    for _ in range(max_iter):
        BT_P_B = B.transpose(-1, -2) @ P @ B
        BT_P_A = B.transpose(-1, -2) @ P @ A
        inv_term = torch.linalg.inv(R + BT_P_B)
        P_new = A.transpose(-1, -2) @ P @ A - A.transpose(-1, -2) @ P @ B @ inv_term @ BT_P_A + Q
        if torch.norm(P_new - P) < eps:
            break
        P = P_new
    return P

class HInfinityLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, q=1e-2, r=1e-2):
        super(HInfinityLSTMCell, self).__init__()
        self.hidden_size = hidden_size

        self.W_i = nn.Linear(input_size, hidden_size)
        self.U_i = nn.Linear(hidden_size, hidden_size)

        self.W_o = nn.Linear(input_size, hidden_size)
        self.U_o = nn.Linear(hidden_size, hidden_size)

        self.W_c = nn.Linear(input_size, hidden_size)
        self.U_c = nn.Linear(hidden_size, hidden_size)

        A = torch.eye(hidden_size)
        B = torch.eye(hidden_size)
        C = torch.eye(hidden_size)
        self.register_buffer('A', A)
        self.register_buffer('B', B)
        self.register_buffer('C', C)

        self.register_buffer('Q', q * torch.eye(hidden_size)) 
        self.register_buffer('R', r * torch.eye(hidden_size)) 


    def forward(self, x, state):
        h_prev, c_prev = state

        P = solve_riccati(self.A, self.B, self.Q, self.R)
        K = torch.linalg.solve(self.R + self.B.T @ P @ self.B, self.B.T @ P @ self.A)

        # Gates
        i = torch.sigmoid(self.W_i(x) + self.U_i(h_prev))
        o = torch.sigmoid(self.W_o(x) + self.U_o(h_prev))
        c_bar = torch.tanh(self.W_c(x) + self.U_c(h_prev))

        I = torch.eye(self.hidden_size, device=x.device)
        c_t = (I - K @ self.C) @ c_prev.unsqueeze(-1) + K @ (i * c_bar).unsqueeze(-1)
        c_t = c_t.squeeze(-1)

        h_t = o * torch.tanh(c_t)

        return h_t, (h_t, c_t)

class HInfinityLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, batch_first=True):
        super(HInfinityLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.cell = HInfinityLSTMCell(input_size, hidden_size)
        self.batch_first = batch_first

    def forward(self, x):
        if self.batch_first:
            batch_size, seq_len, _ = x.size()
        else:
            seq_len, batch_size, _ = x.size()

        h_t = torch.zeros(batch_size, self.hidden_size, device=x.device)
        c_t = torch.zeros(batch_size, self.hidden_size, device=x.device)

        outputs = []
        for t in range(seq_len):
            input_t = x[:, t, :] if self.batch_first else x[t, :, :]
            h_t, (h_t, c_t) = self.cell(input_t, (h_t, c_t))
            outputs.append(h_t.unsqueeze(1))

        return torch.cat(outputs, dim=1)

class CNNHInfinityLSTM(nn.Module):
    def __init__(self):
        super(CNNHInfinityLSTM, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)

        self.dropout = nn.Dropout(0.3)

        self.hinf_lstm1 = HInfinityLSTM(input_size=64 * 32, hidden_size=128, batch_first=True)
        self.hinf_lstm2 = HInfinityLSTM(input_size=128, hidden_size=128, batch_first=True)

        self.fc = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.dropout(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.dropout(x)

        x = x.permute(0, 3, 1, 2)
        x = x.contiguous().view(x.size(0), x.size(1), -1)

        lstm_out1 = self.hinf_lstm1(x)
        lstm_out2 = self.hinf_lstm2(lstm_out1)

        x = self.dropout(lstm_out2[:, -1, :])
        x = self.fc(x)
        return x
