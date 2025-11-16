import os
import glob

folder = r"C:\Users\lenovo\PycharmProjects\Ethereum_Analysis\DataSets"

print("🔍 Checking folder:", folder)
if not os.path.exists(folder):
    print("Folder does not exist!")
else:
    print("Folder found!")

    print("\n📁 Files in DataSets:")
    for f in glob.glob(os.path.join(folder, "*")):
        print("   ", os.path.basename(f))
