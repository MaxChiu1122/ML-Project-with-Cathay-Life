# GitHub 專案管理教學

以下是幾個步驟和建議，幫助你進行 GitHub 專案管理：

## 1. 設置 Git 和 GitHub 帳號
確保每個組員都安裝並設置了 Git，並且有 GitHub 帳號。
- 安裝 Git：提供 Git 的安裝鏈接和安裝指南。
- 設置 Git 用戶名稱和郵箱：
    ```bash
    git config --global user.name "Your Name"
    git config --global user.email "youremail@example.com"
    ```

## 2. 建立專案倉庫
在 GitHub 上創建專案倉庫，並將專案代碼推送到該倉庫。可以使用以下命令來初始化本地倉庫並連接 GitHub 遠端倉庫：
    ```bash
    git init
    git remote add origin https://github.com/MaxChiu1122/ML-Project-with-Cathay-Life.git
    ```

## 3. 分支管理
使用分支管理。這樣每個人可以在自己的分支上工作，避免互相影響。
- 如何創建新分支：
    ```bash
    git checkout -b feature-branch
    ```
- 如何切換到主分支（`main`）並更新：
    ```bash
    git checkout main
    git pull origin main
    ```

## 4. 協作流程（如 GitFlow）
- **拉取請求（Pull Requests, PR）**：讓組員在完成自己分支上的工作後，發送 PR 到主分支。在 PR 中，他們可以描述所做的變更，並要求其他人進行代碼審查。
- **代碼審查和合併**：當 PR 被審核後，通過 PR 將其合併到 `main` 分支。這樣可以確保代碼質量並減少錯誤。

## 5. 定期拉取更新
鼓勵組員定期從 GitHub 上拉取最新的變更，避免版本差異過大：
    ```bash
    git pull origin main
    ```

## 6. 解決衝突
如果發生衝突，請教他們如何處理衝突：
- **衝突情況**：當你從遠端拉取更新時，Git 會告訴你哪些文件有衝突。你需要打開這些文件並手動解決衝突，然後提交解決後的版本。
- **提交衝突解決後的更改**：
    ```bash
    git add .
    git commit -m "Resolved merge conflicts"
    git push origin main
    ```

## 7. 同步開發進度
- 使用 GitHub Issues 或 Project boards 來追蹤進度、分配任務和討論問題。
- 鼓勵組員在推送代碼時提供清晰的提交訊息，並簡單描述變更。

## 8. 最佳實踐
- **頻繁提交**：讓組員養成頻繁提交代碼的習慣，而不是在完成大量工作的時候一次性提交，這樣更容易追蹤問題。
- **有意義的提交訊息**：每次提交都要寫清楚提交的內容，這樣其他組員能夠清楚知道變更的目的。
    例如：
    ```bash
    git commit -m "Fixed bug in data preprocessing script"
    ```

## 9. 處理大型文件
如果專案中包含大型文件，可以介紹 Git LFS（Large File Storage）來處理：
- 安裝 Git LFS 並使用：
    ```bash
    git lfs install
    git lfs track "*.psd"
    ```

reference: https://www.youtube.com/watch?v=FKXRiAiQFiY&t=634s&ab_channel=PAPAYA%E9%9B%BB%E8%85%A6%E6%95%99%E5%AE%A4