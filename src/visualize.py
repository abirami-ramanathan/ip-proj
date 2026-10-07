import os, pandas as pd, matplotlib.pyplot as plt
def main():
    p="results/inference/metrics.csv"
    if not os.path.exists(p): raise SystemExit("Run inference first.")
    df=pd.read_csv(p)
    df.drop(columns=["path"]).groupby("class").mean(numeric_only=True).to_csv("results/inference/class_summary.csv")
    for col in ["MSE","Bias","Variance","RMSE","PSNR","SSIM","MTF_Error","NPS_Error"]:
        ax=df.boxplot(column=col,by="class",rot=30,figsize=(8,5))
        plt.title(col); plt.suptitle(""); plt.tight_layout(); plt.savefig(f"results/inference/{col}.png",dpi=160); plt.close()
    print(df.groupby("class")[["MSE","PSNR","SSIM","MTF_Error","NPS_Error"]].mean())
if __name__=="__main__": main()
