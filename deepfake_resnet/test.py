# 加载 Test 数据目录 + 模型
test_dataset = CelebV2Dataset("/path/to/Celeb_V2/Test", transform)
test_loader = DataLoader(test_dataset, batch_size=32)

# 推理 + 输出准确率