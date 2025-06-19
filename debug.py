# scan_dim.py  用法: python scan_dim.py <pt_root>
import sys, pathlib, torch, collections
root = pathlib.Path(sys.argv[1])
dims = collections.Counter()
bad_files = []
for pt in root.rglob("*.pt"):
    hs = torch.load(pt, map_location="cpu")["hidden_states"]
    dims[hs.size(-1)] += 1
    if hs.size(-1) != 15360:      # 预期维度
        bad_files.append(pt)
print("维度分布:", dims)
print(f"异常文件 {len(bad_files)} 个")
if bad_files:
    print("示例:", bad_files[:5])
