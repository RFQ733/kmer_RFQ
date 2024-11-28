import itertools
import numpy as np
# 定义 RNA 片段的组成部分
rna_bases = ['A', 'U', 'G', 'C', 'T']

# 生成所有长度为 1, 2, 3 和 4 的 RNA 片段
rna_fragments = []
for length in range(1, 5):
    rna_fragments.extend([''.join(fragment) for fragment in itertools.product(rna_bases, repeat=length)])

# 为每个 RNA 片段分配一个np.random 100 维的向量
# # 
rna_dict = {fragment: np.random.rand(5) for fragment in rna_fragments}
# rna_dict = {fragment: idx for idx, fragment in enumerate(rna_fragments)}

# 打印结果
for fragment, idx in rna_dict.items():
    print(f'{fragment}: {idx}')


np.save('rna_dict.npy', rna_dict)