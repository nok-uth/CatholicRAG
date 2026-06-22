@echo off
echo ============================================
echo  CatholicRAG - Git Setup and Initial Commit
echo ============================================

cd /d "C:\Users\rylan\CS_PROJECTS\CatholicRAG"

echo [1/4] Cleaning up any broken .git directory...
if exist ".git" (
    rmdir /s /q ".git"
    echo     Removed old .git folder.
)

echo [2/4] Initializing git repository...
git init -b main
git config user.email "rylan5059@gmail.com"
git config user.name "Rylan"

echo [3/4] Staging files (respecting .gitignore)...
git add .
echo     Staged files:
git status --short

echo [4/4] Making initial commit...
git commit -m "Initial commit: CatholicRAG with Pinecone + OpenAI"

echo.
echo ============================================
echo  Done! Git repo initialized and committed.
echo  Next: add remote and push to GitHub.
echo ============================================
pause
