# 可学习搜索细胞命运因子

- 链接: https://claugpt.xyz/c/6a9171d7-cdc0-83ea-aac5-cc8eaa79f778
- 时间: 2026/8/31 14:35:12

---

思考了 11s
思考了 1m
思考了 14s
可以。为了把这件事真正讲清楚，我建议先暂时不要把它叫“多步最优运输”，也不要一开始就上强化学习。
我会把你的问题重新定义成一句非常具体的话：
学习一个“细胞状态 + 基因干预 → 新细胞状态”的模型，然后从 Th2 出发，在最多 k 个基因干预内，寻找最可能把整个细胞群推向 Th17、同时干预数尽量少的方案。
这实际上是一个学习细胞状态转移规律 + 有约束的规划搜索问题。
我认为最合理的发展路线是：
最基本版：扰动方向库 → 单步评分 → 贪心多步搜索
↓
实用版：状态依赖的扰动模型 → 多步模拟 → 束搜索
↓
最终版：细胞群分布转移模型 → 不确定性感知 → 多步最优规划 → 实验反馈闭环
下面我从最简单的版本开始讲，而且会解释为什么每一步都需要升级。
一、先把问题抽象清楚
假设你现在有两个细胞群：
X
S
​
={x
1
​
,x
2
​
,…,x
n
​
}
是 Th2。
目标细胞群：
X
T
​
={y
1
​
,y
2
​
,…,y
m
​
}
是 Th17。
每个 x
i
​
原来可能是两三万个基因的表达量。
我们先用一个编码器把它压缩：
z
i
​
=E(x
i
​
)
例如把两万个基因变成 50 维。
于是：
z
i
​
∈R
50
可以把这个 z 理解成一个细胞的“状态坐标”。
二、你真正想学习的东西是什么？
其实核心只有一个函数：
z
′
=F(z,a)
其中：
z：干预前的细胞状态；
a：一个动作，例如敲低基因 A；
z
′
：干预后的细胞状态。
也就是说：
如果一个处于状态 z 的 CD4 T 细胞，把基因 A 敲低，它会变成什么状态？
Perturb-seq 的价值就在这里。
它提供大量这样的信息：
(细胞状态,基因干预)⟶干预后状态
一旦这个 F 学得足够好，后面的事情就变成一个搜索问题。
例如：
Th2
KD A
​
S
1
​
KD B
​
S
2
​
KD C
​
Th17
我们需要找到 A、B、C。
三、最基本版：先不要训练复杂模型
这是我最建议你首先实现的版本。
因为它非常简单，却能回答一个重要问题：
你的这个研究方向本身到底有没有信号？
如果这个版本都完全没有信号，直接做复杂模型通常没有意义。
1. 从 Perturb-seq 建立“动作字典”
假设公开数据里有：
对照细胞；
A 敲低细胞；
B 敲低细胞；
C 敲低细胞；
……
先把所有细胞映射到同一个潜在空间。
例如：
z=E(x)
然后计算对照细胞的平均状态：
μ
0
​
=
N
0
​
1
​
i=1
∑
N
0
​
​
z
i
​
对于基因 A：
μ
A
​
=
N
A
​
1
​
i=1
∑
N
A
​
​
z
i
​
那么最简单的 A 的“动作”就是：
Δ
A
​
=μ
A
​
−μ
0
​
同样：
Δ
B
​
=μ
B
​
−μ
0
​
Δ
C
​
=μ
C
​
−μ
0
​
于是我们获得了一个动作库：
A={Δ
A
​
,Δ
B
​
,Δ
C
​
,…}
直觉上可以把它想象成很多箭头。
比如二维示意：
Th17
●
↗
B ↗
●
↗
A ↗
●
Th2
基因 A 的敲低对应一根箭头，基因 B 又对应另一根箭头。
四、然后把 Th2 放进来
你的 Th2 平均状态是：
μ
S
​
Th17 平均状态是：
μ
T
​
如果对 Th2 使用 A，那么最简单的预测就是：
μ
^
​
A
​
=μ
S
​
+Δ
A
​
然后问：
这个新状态是不是比原来的 Th2 更像 Th17？
定义距离：
D(μ
S
​
,μ
T
​
)
干预 A 后：
D(μ
S
​
+Δ
A
​
,μ
T
​
)
如果：
D(μ
S
​
+Δ
A
​
,μ
T
​
)<D(μ
S
​
,μ
T
​
)
说明 A 至少把 Th2 往 Th17 的方向推了一些。
可以定义一个非常直观的分数：
S(A)=D(μ
S
​
,μ
T
​
)−D(μ
S
​
+Δ
A
​
,μ
T
​
)
于是：
S(A)>0：朝 Th17 靠近；
S(A)<0：离 Th17 更远；
S(A) 越大：越值得实验。
这样你马上就得到：
候选基因        分数
A               0.82
F               0.71
C               0.63
K               0.51
...
这就是最基本的候选因子预测器。
五、怎么从一个基因扩展到 k 个基因？
最简单就是贪心搜索。
假设：
k=3
第一步：
从几千个基因里面找：
a
1
∗
​
=arg
a
min
​
D(μ
S
​
+Δ
a
​
,μ
T
​
)
假设找到 A。
于是：
z
1
​
=μ
S
​
+Δ
A
​
第二步继续：
a
2
∗
​
=arg
a
min
​
D(z
1
​
+Δ
a
​
,μ
T
​
)
假设得到 B。
然后：
z
2
​
=z
1
​
+Δ
B
​
第三步再找 C。
最终得到：
Th2
A
​
z
1
​
B
​
z
2
​
C
​
z
3
​
≈Th17
所以最基础算法其实十几行核心代码就能实现。
六、但这个模型有一个巨大的问题
它隐含了一个非常强的假设：
基因 A 的作用在所有细胞状态下都一样。
也就是：
Δ
A
​
(z)=Δ
A
​
这通常不成立。
举个非常直观的例子。
假设基因 A 敲低：
在原始 CD4 T 细胞中：
● ─────A────→ ●
但是在 Th2 中可能：
● ──A──→ ●
在一个已经部分 Th17 化的细胞中甚至可能：
● ←──A── ●
因为基因调控网络依赖细胞状态。
这也是 CellOracle
非常重要的思想之一：扰动的作用是细胞状态依赖的。它通过状态特异性的基因调控网络模拟转录因子扰动，并且已经在造血和发育系统中验证过这一思想。
Nature
所以真正值得做的版本必须升级。
七、实用版：不再学习“箭头”，而是学习“箭头生成器”
这是我认为你的项目真正应该做到的第一版模型。
不再规定：
z
′
=z+Δ
A
​
而学习：
z
′
=F
θ
​
(z,A)
或者写成：
z
′
=z+Δ
θ
​
(z,A)
注意这个变化非常重要。
以前：
Δ
A
​
是固定的。
现在：
Δ
θ
​
(z,A)
依赖当前细胞状态。
于是同一个基因 A：
Δ
θ
​
(z
Th2
​
,A)
和：
Δ
θ
​
(z
intermediate
​
,A)
可以完全不同。
八、Perturb-seq 在这里怎么训练它？
假设某个 Perturb-seq 数据里有很多状态相似的细胞。
对于某个状态区域：
control:
● ● ● ● ●
KD A:
● ● ●
● ● ●
我们希望模型学会：
F
θ
​
(z,A)
能把对照分布变成 A 敲低后的分布。
这其实和 CellOT 的思想很接近：单细胞实验往往不能观察“同一个细胞干预前和干预后”，因为测 RNA 时细胞已经被破坏了，所以我们实际上拥有的是两个未配对的分布。CellOT 正是使用最优运输学习这种扰动前分布到扰动后分布的映射。
Nature
所以这里非常适合借鉴最优运输。
但注意：
最优运输是帮助你学习一次扰动如何改变一个分布，而不是整个问题本身。
这是一个很重要的区别。
九、所以这里其实有两个模型
我会把你的系统拆成：
模型一：状态编码器
z=E(x)
负责回答：
这个细胞现在在哪里？
模型二：扰动转移模型
z
′
=F
θ
​
(z,a)
负责回答：
在这个位置执行动作 a，下一步在哪里？
于是整个系统变成：
Perturb-seq
│
▼
学习扰动转移模型 F
│
▼
Th2 ──→ 当前状态 ──→ 尝试基因 A
│      基因 B
│      基因 C
│      ...
▼
预测新状态
│
▼
与 Th17 比较
│
▼
选择最佳动作
│
▼
进入下一状态
│
...
这就已经非常接近你最开始想象的东西了。
十、为什么这时候不能再简单贪心？
假设有三个基因：
A、B、C。
从 Th2 出发：
A
Th2 ─────────────────→ 看起来离Th17很近
B
Th2 ─────→ 中间状态
│
│ C
▼
Th17
如果只看第一步：
S(A)>S(B)
所以贪心算法一定选择 A。
但是 A 后面可能无路可走。
而 B 虽然第一步没有特别好，却给 C 创造了条件。
于是：
D(F(F(S,B),C),T)<D(F(S,A),T)
这种情况下：
B→C
才是真正的最佳方案。
这正是规划问题出现的地方。
十一、所以实用版应该用“束搜索”
我反而不建议你一开始用强化学习。
因为你的动作空间虽然大，但 k 很可能不会特别大，例如：
k=2,3,4
这时候束搜索非常合适。
假设第一次有 5000 个候选基因。
模型预测：
F(S,A
1
​
),F(S,A
2
​
),...,F(S,A
5000
​
)
保留最好的 50 个。
然后：
Th2
├── A
├── B
├── C
├── D
...
└── top 50
第二步每个再尝试候选动作。
得到：
Th2
├── A ── B
│     ├─ C
│     └─ D
│
├── B ── A
│     ├─ C
│     └─ F
...
再保留最好的 50 条路径。
一直做到 k 步。
最终得到的不是单个基因，而是：
A→C→F
或者：
B→D
这样的候选策略。
十二、而且顺序本身可能成为一个有趣问题
如果转移模型是状态依赖的：
F(F(z,A),B)

=F(F(z,B),A)
那么：
A→B
和：
B→A
理论上可能不同。
为什么？
因为 A 先改变了细胞状态，B 是在新的调控网络背景下起作用。
例如：
A → B：
Th2 ──A──→ state 1 ──B──→ Th17
B → A：
Th2 ──B──→ state 2 ──A──→ other state
这个问题非常有意思。
不过这里有一个实验学上的关键限制：
如果动作是永久性 CRISPR 敲除，那么所谓“多步顺序”必须非常谨慎定义。
因为：
A→B
最后其实是 A、B 都被敲掉。
因此你必须区分：
组合干预问题
{A,B}
和真正的：
序列干预问题
A@t
1
​
→B@t
2
​
后者最好有时间控制的干预体系，否则计算上预测“顺序”很漂亮，实验上却不一定有对应意义。
十三、最终版：不要把一个细胞当状态，而把“一群细胞”当状态
这是我认为最有潜力形成真正方法学工作的地方。
因为你的问题其实不是：
把一个 Th2 细胞变成一个 Th17 细胞。
真正的问题是：
让 Th2 群体的状态分布变成 Th17 群体的状态分布。
Th2 本身不是一个点：
Th2:
●
● ● ●
● ● ● ●
● ●
Th17 也不是：
Th17:
● ●
● ● ●
● ● ●
所以状态应该定义成一个分布：
P
t
​
(z)
而不是一个点 z
t
​
。
十四、最终版的转移模型
于是模型升级成：
P
t+1
​
=F
θ
​
(P
t
​
,a
t
​
)
含义是：
给一群处于分布 P
t
​
的细胞执行基因干预 a
t
​
，预测下一步整个细胞群的分布。
目标是：
P
k
​
≈P
Th17
​
于是优化问题就非常漂亮了：
a
1
​
,…,a
k
​
min
​
D(P
k
​
,P
Th17
​
)
满足：
P
t+1
​
=F
θ
​
(P
t
​
,a
t
​
)
并且：
k≤K
十五、这时候最优运输就真正派上用场了
例如可以用 Wasserstein 距离：
W(P
k
​
,P
T
​
)
衡量：
当前预测出来的一群细胞，与真实 Th17 群体之间还差多少。
目标就是：
a
1:k
​
min
​
W(P
k
​
,P
T
​
)
但我会再加入几个非常重要的项。
十六、最终目标函数
我会设计成：
J=D(P
k
​
,P
T
​
)+λ
1
​
N
pert
​
+λ
2
​
C
tox
​
+λ
3
​
U+λ
4
​
C
OOD
​
​
逐个解释。
第一项：变得像不像 Th17
D(P
k
​
,P
T
​
)
可以使用 Wasserstein 距离、最大均值差异等。
第二项：不要乱敲一堆基因
N
pert
​
也就是干预数量。
例如：
A+B 可以完成，就不要 A+B+C+D。
这非常符合你的实验目标。
第三项：惩罚毒性
C
tox
​
因为某个基因可能让剩下的细胞看起来特别像 Th17，但其实大量其他细胞都死了。
这显然不是我们要的“重编程”。
第四项：不确定性
这是我认为最终模型里非常重要的一项。
假设：
方案A：
预测成功率 90%
模型非常确定
方案B：
预测成功率 97%
模型极度不确定
我宁愿实验 A。
因此：
U(a
1
​
,…,a
k
​
)
表示模型不确定性。
最终模型会主动避开：
“我其实完全没见过这种状态，但我大胆预测它效果特别好”
这样的方案。
第五项：分布外惩罚
如果模型训练时只见过：
naive CD4
activated CD4
memory CD4
现在多步模拟跑到了一个奇怪区域：
×
× ×
×
训练数据 ●●●●
那么后面的预测不能信。
因此加入：
C
OOD
​
即分布外惩罚。
十七、最终版其实就是一个“细胞导航系统”
这是我觉得最容易理解这个方法的比喻。
Perturb-seq 相当于告诉模型：
在不同地方按下不同基因按钮，细胞通常会往哪里走。
于是学习：
F
θ
​
(P,a)
相当于学习一张地图。
你的 Th2 是：
起点。
Th17 是：
终点。
基因敲低是：
可以选择的道路。
你的算法负责寻找：
Th2→A→B→C→Th17
而不是：
哪个基因和 Th17 最相关？
这是你的想法和普通差异表达分析非常本质的区别。
十八、举一个完整的假想例子
假设只允许三个候选动作：
A、B、C。
最大：
k=2
模型首先模拟。
单步
S
A
​
P
A
​
距离 Th17：
D(P
A
​
,T)=0.40
B：
D(P
B
​
,T)=0.55
C：
D(P
C
​
,T)=0.70
所以单基因来看：
A>B>C
传统方法可能直接告诉你：
做 A。
但是你的模型继续搜索。
两步
发现：
D(P
A→B
​
,T)=0.30
D(P
A→C
​
,T)=0.28
而：
D(P
B→C
​
,T)=0.08
突然发现：
B→C
远好于 A。
原因可能是 B 首先把 Th2 推到了一个中间状态：
Th2 ─────→ intermediate ─────→ Th17
B                    C
虽然 B 单独看并不起眼。
这就是多步规划真正可能比普通 perturbation ranking 强的地方。
十九、GEARS 这类方法和你的区别在哪里？
GEARS
已经证明了一个很重要的事情：
可以利用单细胞扰动数据和基因关系图，预测未实验过的多基因组合造成的转录结果。
它能够预测单基因和多基因扰动，并针对未实验过的组合进行泛化。
Nature
所以你不能把论文贡献简单定义成：
“我预测多个基因组合。”
这已经有人做了。
你的问题更适合定义成：
给定任意起始细胞状态 S 和目标状态 T，自动寻找最小、可靠的干预集合或干预策略，使整个细胞状态分布从 S 转向 T。
这就从：
扰动预测
变成了：
目标导向的逆向干预设计。
这是非常重要的区别。
二十、为什么我暂时不推荐强化学习
从数学形式上，你这个问题确实非常像：
状态→动作→状态→奖励
完全可以写成马尔可夫决策过程。
奖励：
R
t
​
=−D(P
t
​
,P
T
​
)
然后训练策略：
π(a∣P)
理论上很漂亮。
但你最大的瓶颈不是“不会选择动作”。
而是：
你到底能不能准确预测执行一个动作之后细胞会去哪？
也就是：
F
θ
​
是否可靠。
如果：
F
θ
​
不准，强化学习只会特别聪明地利用模型错误。
所以研究顺序应该是：
先把世界模型学好→再做规划
​
而不是：
先训练一个复杂策略
二十一、我建议你的实际开发路线
如果这个项目由我来做，我会非常明确地分四代。
版本	核心	搜索	目的
V0	固定扰动向量	单基因/贪心	验证有没有信号
V1	状态依赖转移模型	束搜索	做真正的多步预测
V2	学习组合/非线性作用	束搜索/模型预测控制	解决基因互作
V3	群体分布 + 不确定性	不确定性感知规划	最终模型
我尤其建议一定做 V0。
因为 V0 会成为以后整篇工作的一个非常重要的基线。
如果 V3 最后只比：
z
′
=z+Δ
a
​
好一点点，那说明复杂模型价值有限。
反过来，如果：
V0<V1<V2<V3
非常清楚，那整个故事就漂亮了。
二十二、还有一个非常重要的地方：目标不能只定义成“转录组像 Th17”
最终我不会只使用：
D(P,P
Th17
​
)
而会至少有两个目标。
一个是全局状态：
D
global
​
另一个是 Th17 功能程序：
D
program
​
所以：
D=αD
global
​
+βD
program
​
因为存在一种危险情况：
预测细胞
↓
转录组整体很像 Th17
↓
但关键功能程序没有建立
这不能叫真正的 Th2 → Th17 转化。
因此最终实验验证也不能只做单细胞测序。
最好还验证：
Th17 核心转录程序；
蛋白/细胞因子；
必要时功能性表型；
细胞活性和增殖。
二十三、这个项目最难的，其实不是搜索算法
我反而觉得这里最容易走偏。
真正困难程度大概是：
可靠的扰动转移模型>跨数据集迁移>组合效应>搜索算法
​
搜索用束搜索都可以做得很好。
真正决定项目成败的是：
F
θ
​
(z,a)
能不能从公开 Perturb-seq 学到可迁移、状态依赖的因果扰动规律。
CellOT 已经展示了用最优运输从未配对的对照/扰动细胞分布学习单细胞响应映射的路线；CellOracle 展示了状态依赖的调控网络扰动；GEARS 展示了组合扰动预测。因此你的机会并不是重新发明其中任何一个模块，而是把问题提升成目标状态驱动的多步干预设计。
Nature
+2
Nature
+2
二十四、我现在最推荐的“第一篇可实现方案”
如果让我现在替你把方法定下来，我不会直接做最终豪华版本，而是：
共享潜在空间+状态依赖扰动模型+分布距离+束搜索+不确定性过滤
​
具体就是：
X
E
​
P(z)
然后：
P
t+1
​
=F
θ
​
(P
t
​
,a
t
​
)
搜索：
a
1:k
∗
​
=arg
a
1:k
​
min
​
[D(P
k
​
,P
T
​
)+λk+γU(a
1:k
​
)]
其中：
k≤K
输出不要只有一个答案，而是：
候选方案          预测转化效果     不确定性     干预数
A                 中              低           1
B + C             高              低           2
D + F             很高            高           2
A + G + H         很高            中           3
然后算法推荐：
第一优先实验：B+C
第二优先实验：A
第三优先实验：A+G+H
而不是机械地选择预测分数最高的 D+F，因为它的不确定性太高。
这已经是一个非常完整、可实现、也和你的实验资源约束高度匹配的系统。
而最终版再进一步变成：
可迁移的状态依赖扰动算子+群体分布动力学+组合基因互作+不确定性感知规划+实验反馈闭环
​
我认为这比单纯把问题包装成“多步最优运输”更准确，也更有发展空间。
下一步其实最值得深入的已经不是继续泛泛谈框架，而是把 F
θ
​
到底怎么训练彻底设计出来：Perturb-seq 没有真正的“同一个细胞干预前后配对”，那么训练样本怎么构造、最优运输怎么嵌进去、怎样让模型从普通 CD4 状态迁移到 Th2、损失函数具体写成什么，以及怎样防止模型在第二三步开始胡乱外推。这一部分是整个项目最核心的方法学问题。
把 Fθ​ 到底怎么训练彻底设计出来： 我之前记得看到过一个STATE模型，就是用来学习这个的，你可以搜索并阅读文章或github代码 ，确定是否能够应用在我们的项目中，或者有没有更好的可以直接使用的模型。
思考了 1m 34s
我查完之后，结论比我们前面讨论时更明确了，而且有一个非常重要的新情况：
STATE 确实就是我们想要的 F
θ
​
这一类模型，而且可以直接用于这个项目。
但截至 2026 年 8 月，我不建议“只拿 STATE 直接套”。因为现在已经出现了两个对我们更关键的东西：
一是 Stack，专门解决“目标细胞状态没有 perturbation 数据”的 zero-shot 问题；二是刚刚发表的、几乎为我们这个项目量身定做的 2200 万 primary human CD4+ T-cell genome-scale Perturb-seq。
更关键的是，这个 CD4 Perturb-seq 的作者自己已经做了一个 pert2state_model，直接用 perturbation signatures 去重构 Th1/Th2 等目标细胞状态。也就是说，我们最基础的 V0 甚至不用自己重新发明。
科学直通车
+1
所以，我现在会把整个方案调整成：
pert2state V0→STATE/CellFlow 学 F
θ
​
→Stack 做 zero-shot 校准→多基因规划
​
而不是从 K562/RPE1 的 Perturb-seq 开始硬迁移到 Th2。
1. 先说 STATE：它确实就是我们要找的东西
你记得的应该就是 Arc Institute 的 State。
STATE 官方 GitHub
STATE 原始论文
STATE 实际上有两个部分：
STATE=SE+ST
SE 是 State Embedding，负责把细胞转成一个相对稳定的状态表示。
ST 是 State Transition，负责：
一群 control cells+perturbation→一群 predicted perturbed cells
​
这和我们之前定义的：
P
t+1
​
=F
θ
​
(P
t
​
,a
t
​
)
几乎完全一致。
而且 STATE 最重要的地方是，它不是简单学习：
Δ
g
​
=μ
g
​
−μ
0
​
而是让一个 Transformer 同时观察一组 control cells，利用细胞之间的 self-attention 建模异质性，再预测 perturbation 后的整个 cell set。论文中特别强调，它的目标就是利用 cellular context 来提高 perturbation effect 在不同 context 之间的泛化。
开放评审
+1
所以从概念上说：
STATE 的 ST≈我们一直说的 F
θ
​
​
2. STATE 的一条训练数据到底是什么？
这是我专门去看代码以后觉得非常适合我们的地方。
STATE 配套的 cell-load 已经把数据接口写得比较完整。它读取 AnnData，并且明确区分：
pert_cell_emb、ctrl_cell_emb、pert_emb、pert_name、cell_type、batch 等字段；control 还可以按 batch 或随机策略从匹配 context 中抽样。
GitHub
STATE 的 cell-load 数据接口
也就是说，它并不要求：
x
i
control
​
↔x
i
perturbed
​
是真正同一个细胞。
这是非常重要的，因为 Perturb-seq 根本不可能获得真正的：
同一个细胞干预前 → 测 RNA → 再干预后测 RNA。
STATE 做的是：
X
0
​
={x
1
ctrl
​
,...,x
m
ctrl
​
}
和：
X
g
​
={x
1
g
​
,...,x
m
g
​
}
来自相同实验 context，但不需要 cell-to-cell 配对。
训练：
X
^
g
​
=F
θ
​
(X
0
​
,g,c)
然后让：
X
^
g
​
∼X
g
​
也就是预测出来的分布接近真实 perturbation 分布。
这正适合我们的数据结构。
3. 它不是在学习“一个细胞对应另一个细胞”
这是 STATE 和普通回归非常重要的区别。
例如有：
control CD4:
● ● ● ● ● ● ● ●
gene A CRISPRi:
● ●
● ● ●
● ●
STATE 不会强行说：
control cell #17 → perturb cell #43
而是学习：
F
θ
​
({control population},A)→{perturbed population}
所以我们真正得到的是：
P(X
′
∣X,g)
而不是：
x
′
=x+Δ
g
​
这已经比我们前面设计的固定 perturbation vector 强很多。
STATE 的理论部分甚至专门证明了，它的解空间可以包含连续最优运输映射，因此可以把它理解成一种比经典 OT 更灵活的 distribution-to-distribution perturbation model。
BioRxiv
4. STATE 是否支持基因 KD/KO？
支持。
论文不只是药物数据，还专门在 Replogle–Nadig genetic perturbation 数据上训练过，包含约 2,024 个遗传扰动、四个细胞系。官方 GitHub 现在甚至直接提供：
“Train an ST model for genetic perturbation prediction using the Replogle-Nadig dataset”
的训练入口。
GitHub
+1
因此我们的：
a=CRISPRi gene A
完全属于 STATE 的设计范围。
而且 cell-load 支持两种 perturbation representation。
最简单的是：
e
g
​
=one-hot(g)
也可以通过 perturbation_features_file 给每个基因一个预计算 embedding。
GitHub
后者对我们更重要，因为理论上可以变成：
e
g
​
=GeneEmbedding(g)
从而让相似功能的基因拥有相似 representation。
不过这里要注意一个问题：
默认 one-hot STATE 并不能真正预测训练中完全没出现过的新基因。
2026 年 MORPH 的独立比较也明确指出：使用 one-hot perturbation identity 时，STATE 无法编码训练 vocabulary 中不存在的 perturbation，因此真正的 “unseen gene + unseen cell state” 双重 zero-shot 是它的弱点之一。
BioRxiv
5. 但是我发现了一个比 STATE 本身更重要的东西
这是这次检索最大的收获。
就在 2026 年 8 月 28 日，Cell 上线了一篇：
Genome-scale perturb-seq in primary human CD4+ T cells maps context-specific regulators of T cell programs and human immune traits
这不是 K562。
不是 RPE1。
不是癌细胞系。
而是：
primary human CD4
+
T cells
​
而且规模非常大：
∼22million cells
来自：
4 donors
并且包含：
Rest,Stim8hr,Stim48hr
三个状态，对所有表达基因进行了 genome-scale CRISPRi。
科学直通车
+1
Primary Human CD4+ T Cell Perturb-seq 官方数据集
这件事情会直接改变我们项目的数据策略。
之前我们的问题是：
K562 perturbation→?Th2 perturbation
domain shift 非常大。
现在变成：
primary human CD4 perturbation→Th2
虽然仍然有 cell-state shift，但已经是同一主要细胞谱系了。
这好很多。
6. 更巧的是：作者已经做了一个非常接近我们 V0 的模型
这个东西甚至名字都很直接：
pert2state_model 官方 GitHub
它的 README 写的就是：
reconstructing target state signatures from perturbation effects.
而且论文明确说，他们已经利用这个 genome-scale CD4 Perturb-seq 的 perturbation signatures，去解释自然人群单细胞图谱里的 T-cell states，并且用于寻找 Th1/Th2 polarization 的正负调控因子。
科学直通车
+1
这跟我们的：
Th2→Th17
已经非常接近了。
7. 我把它的代码也看了一下，原理反而非常简单
Perturb2StateModel.py 的核心是 Elastic Net。
代码里定义：
X
是：
X∈R
G×P
其中：
G：基因；
P：perturbations。
每一列：
X
⋅,g
​
就是：
敲低基因 g 后，全转录组发生的变化。
然后：
y∈R
G
是目标 physiological state 的 differential expression signature。源码的 fit() 就明确要求 X 是 known perturbations 的 differential-expression estimates，y 是 physiological/unknown condition 的 differential-expression estimates；模型用 ElasticNetCV 做稀疏回归。
GitHub
+1
于是它实际上求的是：
y≈β
1
​
Δ
1
​
+β
2
​
Δ
2
​
+⋯+β
P
​
Δ
P
​
​
同时通过 Elastic Net 让大部分：
β
g
​
=0
最后只剩几个基因。
这跟我们最早提出的：
用尽量少的 perturbation 把 source→target
已经非常接近。
8. 对 Th2 → Th17，我们可以马上这样做
定义：
y=μ
Th17
​
−μ
Th2
​
这是我们真正希望实现的转录组变化。
然后从 CD4 Perturb-seq 得到：
X=[Δ
g
1
​
​
,Δ
g
2
​
​
,…,Δ
g
P
​
​
]
求：
β
min
​
∥y−Xβ∥
2
2
​
+λ
1
​
∥β∥
1
​
+λ
2
​
∥β∥
2
2
​
最后可能得到：
y≈0.72Δ
A
​
+0.41Δ
B
​
+0.19Δ
C
​
于是：
Th2 → Th17
主要候选：
A KD
B KD
C KD
这个东西我建议一定做。
因为它已经有作者代码，而且同一篇论文已经在 primary CD4 T-cell polarization 上证明这种思想有用。
科学直通车
+1
但它不是我们的最终模型。
因为：
Δ
A
​
仍然基本上是固定 perturbation signature。
它没有真正解决：
Δ
A
​
(Th2)

=Δ
A
​
(intermediate)
这个问题。
9. 所以 pert2state 和 STATE 正好是两个层次
这两个模型千万不要混在一起。
模型	实际解决的问题	对我们的作用
pert2state	哪些 perturbation signatures 能重构目标状态	V0，直接筛候选基因
STATE/ST	当前细胞群 + perturbation → perturb 后细胞群	真正的 F
θ
​
CellFlow	条件化 distribution → distribution generative transport	非常适合最终 F
θ
​
Stack	从其它 context 的例子直接 zero-shot 推断新 context	解决 Th2 没有 perturbation 数据
GEARS/TxPert	预测 unseen gene / combination response	组合与 unseen perturbation 辅助模块
其中 CellFlow 是我这次重新评估后觉得非常值得和 STATE 并行测试的模型。
CellFlow 官方 GitHub
它直接定义：
P
pert
​
=F
θ
​
(P
control
​
,c)
底层使用 flow matching + neural optimal transport，并且官方明确支持 genetic modification、cytokine、drug、组合 perturbation，甚至 cell-fate engineering。它的 Python API 也是直接接受 AnnData，然后 predict() 生成 perturbation response。
BioRxiv
+1
从“我们的最终数学问题”来说：
CellFlow 比 STATE 更像我原本想设计的 F
θ
​
​
但从“现成大规模工程成熟度”来说：
STATE 更成熟
​
因为 STATE 已经有完整 genetic perturbation pipeline、cell-load、模型权重和大规模训练经验，而 CellFlow 当前更像一个很漂亮、很灵活的研究框架。
10. 还有一个非常关键的新模型：Stack
这是 STATE 团队 2026 年的新模型。
Stack 官方 GitHub
Stack 论文
它解决的恰恰是 STATE 对我们最麻烦的问题：
如果 Th2 本身没有做过 Perturb-seq 怎么办？
Arc Institute 自己现在对两者的定位非常明确：
STATE 更适合：
已经有比较大的 perturbation dataset，希望扩展这个实验体系。
Stack 更适合：
新 donor、新 disease state、新 tissue、新 cell context，没有这个 context 下的 perturbation data，希望直接 zero-shot 推断。
亚文化研究所
这跟我们的情况高度吻合。
我们可能拥有：
public CD4 Perturb-seq:
control CD4 → gene A KD
control CD4 → gene B KD
control CD4 → gene C KD
以及自己的：
Th2 cells
Th17 cells
但没有：
Th2 + gene A KD
Stack 的目标正是利用其他 context 的信息，在新的 target context 上做 in-context prediction。新版论文的 Perturb Sapiens 已扩展到 28 tissues、40 cell types、892 种药物、细胞因子和遗传 perturbations。
BioRxiv
所以：
Stack 是我们必须加入的 zero-shot baseline
​
甚至可能最后表现比我们自己训练的 STATE 更好。
11. 那为什么我仍然不建议直接把 Stack 当最终 F
θ
​
？
因为我们的最终问题不是单纯：
gene A 在 Th2 上会怎么样？
而是：
Th2
A
​
S
1
​
B
​
S
2
​
→Th17
我们需要一个可以反复调用、可以进入 planner 的：
F
θ
​
(P,a)
STATE 和 CellFlow 在结构上更适合做这个。
Stack 更适合：
zero-shot oracle / teacher
也就是帮我们判断：
“STATE 说 A 在 Th2 上这样作用，Stack 同不同意？”
如果：
F
STATE
​
和：
F
Stack
​
方向高度一致，我们对这个候选的信心会明显增加。
12. 我现在会怎样真正训练我们的 F
θ
​
我建议先不动 600M 的 SE。
第一版直接：
x∈R
2000
使用 2,000 HVGs，或者 128–512 维 PCA。
这是有原因的：2026 年新的 representation benchmark 发现，高维 PCA 在 perturbation reconstruction 上非常有竞争力；foundation-model embedding 并不是自动就比简单 representation 强。另一个 2026 benchmark 甚至发现 TabPFN/TabICL 这样的通用表格模型，在多个 Perturb-seq benchmark 和 primary CD4 genome-wide screen 上能匹配或超过多种 specialized models。
BioRxiv
+1
所以第一阶段千万不要：
“foundation model 越大越好。”
我们应该真正 benchmark。
13. 我会把一条 STATE 训练样本定义成这样
例如：
c=(donor 1,Stim48hr)
基因：
a=KD gene A
随机抽一组 control：
X
0
​
={x
1
​
,…,x
64
​
}
再抽同 donor、同 stimulation condition 下的 A KD：
X
A
​
={y
1
​
,…,y
64
​
}
于是：
(X
0
​
,A,c)→X
A
​
​
注意：
x
i
​
不需要对应：
y
i
​
。
模型：
X
^
A
​
=F
θ
​
(X
0
​
,e
A
​
,c)
14. 我会比原始 STATE 多加几个 loss
最主要的还是 distribution loss：
L
dist
​
=MMD(
X
^
A
​
,X
A
​
)
让整个预测分布像真实分布。
但我们的目标最终是重编程设计，所以我还会加一个 perturbation-delta loss：
L
Δ
​
=
​
(
X
^
ˉ
A
​
−
X
ˉ
0
​
)−(
X
ˉ
A
​
−
X
ˉ
0
​
)
​
1
​
也就是说：
不仅预测细胞本身要像，还必须把 perturbation 引起的变化方向预测对。
再加入 differential-expression loss：
L
DE
​
=∥
ΔDE
A
​
−ΔDE
A
​
∥
以及 NTC identity loss：
L
ctrl
​
=D(F
θ
​
(X
0
​
,NTC),X
0
​
)
最终：
L=L
dist
​
+λ
1
​
L
Δ
​
+λ
2
​
L
DE
​
+λ
3
​
L
ctrl
​
​
这会比单纯追求 gene-expression reconstruction 更适合我们的 planner。
因为 planner 真正在乎的是：
方向有没有预测对
而不是：
所有 housekeeping genes 有没有预测到小数点后几位
。
15. 训练集怎么切，比模型本身还重要
这里绝对不能 random cell split。
否则同一个：
donor
gene
condition
cell state
的细胞一半进训练、一半进测试，结果会漂亮得毫无意义。
我们真正需要测试的是三件事情：
unseen donor
unseen cell state
以及最困难的：
unseen state + perturbation
特别是我们最终需要的是：
在没有 Th2 perturbation 数据时预测 Th2 response
​
STATE 原论文最主要的 genetic benchmark 其实没有严格做到这一点：Replogle-Nadig 的 held-out cell line 设置中，测试 cell line 的约 30% perturbations 仍进入训练，因此更接近 few-shot context generalization，而不是完全 zero-shot。
BioRxiv
+1
这也是为什么 Stack 对我们这么重要。
16. 还有一个很大的警告：STATE 不能因为名字叫 State Transition，就直接多步 rollout
这是我现在最想强调的一点。
STATE 学到的是：
control
perturbation
​
endpoint
它不是从真正 longitudinal trajectory 学出来的：
t
0
​
→t
1
​
→t
2
​
所以如果我们直接：
S
1
​
=F(S
0
​
,A)
然后：
S
2
​
=F(S
1
​
,B)
再：
S
3
​
=F(S
2
​
,C)
数学上可以跑。
但第二步开始：
S
1
​
已经可能是模型生成的、训练时没见过的状态。
于是会出现：
rollout error accumulation
更重要的是：
A→B
不能自动解释成真实实验中的：
先敲 A，过一天，再敲 B。
这需要真正 sequential perturbation 数据。
因此第一篇工作我会把它更谨慎地定义成：
multi-perturbation state planning
​
而不是：
真实时间动力学控制
​
。
17. 这次搜索后，我对整个项目的推荐发生了一个重要变化
不要首先从 STATE 开始。
首先利用刚发表的 primary CD4 genome-scale Perturb-seq 做：
Th2→Th17 target signature
​
然后直接运行作者的：
pert2state
​
得到第一批候选基因。
这就是我们的 V0。
而且这里有一个很漂亮的小技巧：如果我们的实验只允许 CRISPRi/KD，那么 Elastic Net 最好限制：
β
g
​
≥0
源码本身就有 positive 参数。
GitHub
+1
因为：
β
A
​
<0
意味着：
“A KD signature 的反方向有利于 Th17。”
生物学解释实际上更接近：
A activation/OE 可能有利。
如果我们只能 KD，它就不应该成为直接候选。
18. 然后才进入真正的 F
θ
​
我会同时训练两个：
F
STATE
​
和：
F
CellFlow
​
数据完全一样。
都是：
(control CD4 population,gene KD,donor,activation context)→perturbed population
然后做严格的：
leave-donor-out
leave-state-out
double-unseen
benchmark。
谁赢就用谁。
我不会预先规定“STATE 一定赢”。
因为最近的 benchmark 已经很清楚地提醒这个领域：复杂 foundation model 并不总是胜过简单 baseline；在 primary CD4 genome-wide perturbation prediction 上，目前所有模型其实都还相当困难。
BioRxiv
19. 最终整个系统我现在会设计成这样
Primary CD4 Perturb-seq
22M cells
│
┌───────────────┴──────────────┐
│                              │
▼                              ▼
pert2state V0                STATE / CellFlow
│                              │
│ candidate genes              │ learn Fθ
▼                              ▼
A B C D E ...                P' = Fθ(P, gene)
│                              │
└──────────────┬───────────────┘
│
▼
用户自己的 Th2
│
┌──────────┼──────────┐
▼          ▼          ▼
KD A       KD B       KD C
│          │          │
▼          ▼          ▼
P_A        P_B        P_C
│          │          │
└──────────┼──────────┘
▼
Stack zero-shot
交叉校验
│
▼
与真实 Th17 比较
│
▼
Beam Search
│
▼
A / B+C / D+E ...
│
▼
uncertainty
+ toxicity
+ OOD
│
▼
小规模实验验证
这个架构我现在认为比我们之前单纯：
Perturb-seq→F
θ
​
→RL
要稳健得多。
20. 最关键的是：我们的论文创新点也要随之调整
因为这篇新的 primary CD4 Perturb-seq 论文已经做了：
perturbation signatures → natural T-cell state → regulators of Th1/Th2 polarization。
科学直通车
所以我们以后不能把创新点写成：
“利用 Perturb-seq 寻找让一种 T-cell state 变成另一种 T-cell state 的基因。”
这个已经不够新了。
真正可以做成我们方法核心的是：
static perturbation signature→state-conditioned counterfactual transition→multi-gene constrained planning
​
也就是从：
y≈
g
∑
​
β
g
​
Δ
g
​
升级到：
P
t+1
​
=F
θ
​
(P
t
​
,a
t
​
)
再解决：
a
1:k
∗
​
=arg
a
1:k
​
min
​
D(P
k
​
,P
Th17
​
)+λk+γU+ηC
tox
​
​
这里才是我们真正可以拉开差距的地方。
我的最终选择
如果现在就让我决定技术路线，我会选：
pert2state + STATE + Stack
​
CellFlow 作为非常重要的竞争模型。
其中 pert2state 不负责 F
θ
​
，而负责快速证明：
“primary CD4 Perturb-seq 里面确实存在可以解释 Th2→Th17 方向的可干预信号。”
STATE 负责真正学习：
F
θ
​
(P,g,c)→P
′
​
Stack 负责我们最危险的环节：
public CD4 context→用户自己的 Th2 context
​
的 zero-shot 检查。
而且我建议第一轮 STATE 不要用 SE-600M，直接做 ST + HVG/PCA；等基线跑通以后，再加 SE、Stack embedding。这样可以非常清楚地知道性能提升究竟来自哪里，而不是一开始把所有 foundation model 堆进去。
最重要的是，我们现在已经不缺训练数据了：刚发布的 primary human CD4 genome-scale Perturb-seq，可能就是这个项目目前最关键的数据资产。
科学直通车
+1
下一步最值得做的是把这个方案进一步落到真正可以开始写代码的程度：我可以直接按照这个 CD4 数据集的实际 AnnData 字段，给你设计 Th2/Th17 → pert2state → STATE training dataset → train/val/test split → loss → inference → beam search 的完整数据流，包括每个 .obs/.var/.obsm 放什么、第一版训练多少基因/多少细胞、STATE 配置怎么改，以及我们到底需不需要下载完整 2200 万细胞。那一步就可以从“讨论模型”正式进入“项目实现”。
你认为具体实施我们目前的计划，包括第一版，最终版等，对显卡的要求是多少，我现在用的VGPU-32GB可行吗，1-2卡，或者单卡5090可以吗
ChatGPT 也可能会犯错。请核查重要信息。
极速