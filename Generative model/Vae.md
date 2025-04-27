VAE的核心思想：VAE是一种生成模型，目标是学习数据的Lantent distribution，并能够生成新的数据样本。

AE:直接学习数据的压缩表示（编码），再通过解码器重建数据，没有显示的建模Lantent的概率分布  
VAE:将输入数据映射到Lantent的概率分布（高斯分布），通过采样生成新的数据

（1）概率生成模型
VAE假设数据由以下过程生成：  
1、从先验分布p（z）种采样一个潜在变量z  
2、通过解码器p_{\theta}(x|z)生成数据x

目标是最大化数据的边际似然$\p_{\theta}(x)$,但是直接计算困难（因为需要积分所有可能的z）

(2)变分推断
VAE引入一个近视后验分布q_{\phi}(z|x)(编码器)来解决计算难题，优化目标是证据下界


\log p_{\theta}(x) \geq \text{ELBO} = \mathbb{E}_{q_{\phi}(z|x)}[\log p_{\theta}(x|z)] - D_{\text{KL}}(q_{\phi}(z|x) \| p(z))

\begin{itemize}
\item 第一项：重建损失（解码器生成的数据与原始数据的相似度）。
\item 第二项：KL散度，约束 $q_{\phi}(z|x)$ 接近先验分布 $p(z)$。
\end{itemize}

{模型结构}

VAE包含两部分：

编码器 (Encoder)

\item 输入数据 $x$，输出潜在变量 $z$ 的分布参数（均值和方差）。
\item 例如：$q_{\phi}(z|x) = \mathcal{N}(\mu_{\phi}(x), \sigma_{\phi}(x))$。
\item 实际实现中，通常用神经网络预测 $\mu$ 和 $\log \sigma^2$。

解码器 (Decoder)
\begin{itemize}
    \item 从潜在空间采样 $z$，生成数据 $x$ 的分布。
    \item 例如：$p_{\theta}(x|z)$ 可以是伯努利分布（二值数据）或高斯分布（连续数据）。
\end{itemize}

重参数化技巧 (Reparameterization Trick)

为了允许梯度反向传播，采样操作通过以下方式实现：
$$
z = \mu_{\phi}(x) + \sigma_{\phi}(x) \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$
这样，随机性由外部变量 $\epsilon$ 引入，模型可以优化 $\mu$ 和 $\sigma$。



训练过程

\begin{enumerate}
    \item \textbf{前向传播：}
        \begin{itemize}
            \item 输入数据 $x \rightarrow$ 编码器输出 $\mu, \sigma \rightarrow$ 采样 $z \rightarrow$ 解码器生成 $x'$。
        \end{itemize}
    
    \item \textbf{损失函数：}
        $$
        \mathcal{L}(\theta, \phi) = -\text{ELBO} = \text{重建损失} + \text{KL散度}
        $$
        \begin{itemize}
            \item 重建损失：例如二元交叉熵（图像生成）或均方误差。
            \item KL散度：闭合解为 $-\frac{1}{2} \sum (1 + \log \sigma^2 - \mu^2 - \sigma^2)$。
        \end{itemize}
    
    \item \textbf{反向传播：} 通过重参数化技巧计算梯度，更新编码器和解码器参数。
\end{enumerate}

## 代码
class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)  # 输出μ和logσ²
        )
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # 假设输入数据在[0,1]
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # 编码
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=-1)
        z = self.reparameterize(mu, logvar)
        # 解码
        x_recon = self.decoder(z)
        return x_recon, mu, logvar

# 损失函数
def loss_function(x_recon, x, mu, logvar):
    recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss

