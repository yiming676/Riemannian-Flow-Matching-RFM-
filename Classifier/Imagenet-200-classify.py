import csv
import os

def check_mode_coverage(csv_path, class_list=None, save_uncovered=True):
    """
    检查预测结果的模式覆盖率
    Args:
        csv_path (str): 预测结果CSV路径
        class_list (list[str] or None): 所有类别名称列表；若为 None 则自动从csv中提取
        save_uncovered (bool): 是否保存未覆盖类别列表到文件
    """

    predicted_labels = set()

    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            predicted_labels.add(row['predicted_label'])

    if class_list is None:
        class_list = sorted(predicted_labels)
        print("[Warning] 未提供所有类别列表，将以预测结果自身为全集进行评估。")

    all_classes = set(class_list)
    uncovered = all_classes - predicted_labels

    print(f"\n=== 模式覆盖率评估 ===")
    print(f"类别总数：{len(all_classes)}")
    print(f"已覆盖类别数：{len(predicted_labels)}")
    print(f"未覆盖类别数：{len(uncovered)}")

    if uncovered:
        print("未覆盖的类别标签：")
        for cls in sorted(uncovered):
            print(f"- {cls}")

        if save_uncovered:
            with open("uncovered_classes.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["uncovered_label"])
                for cls in sorted(uncovered):
                    writer.writerow([cls])
            print("[未覆盖类别已保存至 uncovered_classes.csv]")
    else:
        print("[所有类别已覆盖]")

if __name__ == '__main__':
    result_csv = "submission.csv"

    train_dir = r"D:\\riemannian-fm\\tiny-imagenet-200\\train"
    all_classes = sorted(os.listdir(train_dir))

    check_mode_coverage(result_csv, class_list=all_classes)
