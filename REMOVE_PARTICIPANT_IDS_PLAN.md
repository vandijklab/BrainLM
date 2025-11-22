# Plan to Remove UK Biobank Participant IDs from Repository

## ⚠️ UK Biobank Requirement
**UK Biobank has explicitly requested that we DELETE the repository and create a NEW, CLEAN repository.** This is the required approach to ensure compliance.

## Overview
This document outlines the steps to:
1. Remove participant IDs from notebook outputs
2. Create a completely new repository (no Git history)
3. Delete the old repository as required by UK Biobank
4. Push the clean code to the new repository

## Affected Files
The following notebooks contain participant IDs in their cell outputs:
1. `inference_04_cls_token_svm_regression.ipynb` - ID: REDACTED_PARTICIPANT_ID
2. `inference_04_cls_token_mlp_regression.ipynb` - ID: REDACTED_PARTICIPANT_ID
3. `inference_03_cls_token_mlp_classification.ipynb` - ID: REDACTED_PARTICIPANT_ID
4. `inference_02_cls_token_knn_regressor.ipynb` - ID: REDACTED_PARTICIPANT_ID
5. `inference_01_cls_token_raw_data_plotting.ipynb` - ID: REDACTED_PARTICIPANT_ID
6. `toolkit/BrainLM_Tutorial.ipynb` - ID: REDACTED_PARTICIPANT_ID

## Step-by-Step Plan

### Phase 1: Fix Current Notebooks (Safe, Non-Destructive)

**Goal**: Remove cell outputs while keeping code intact

1. **Clear outputs from affected notebooks**
   ```bash
   # Using jupyter nbconvert to clear outputs
   jupyter nbconvert --clear-output --inplace inference_04_cls_token_svm_regression.ipynb
   jupyter nbconvert --clear-output --inplace inference_04_cls_token_mlp_regression.ipynb
   jupyter nbconvert --clear-output --inplace inference_03_cls_token_mlp_classification.ipynb
   jupyter nbconvert --clear-output --inplace inference_02_cls_token_knn_regressor.ipynb
   jupyter nbconvert --clear-output --inplace inference_01_cls_token_raw_data_plotting.ipynb
   jupyter nbconvert --clear-output --inplace toolkit/BrainLM_Tutorial.ipynb
   ```

   **Alternative**: Use Python script to programmatically remove outputs
   - This preserves code cells but removes all outputs
   - Notebooks will still be functional (users can re-run cells)

### Phase 2: Create New Clean Repository (No Git History)

**Goal**: Create a completely fresh repository with no Git history

**Steps**:

1. **Create a new directory for the clean repository**
   ```bash
   cd /Users/david/Projects
   mkdir BrainLM-clean
   cd BrainLM-clean
   ```

2. **Copy all files EXCEPT .git directory**
   ```bash
   # Copy everything except .git
   rsync -av --exclude='.git' /Users/david/Projects/BrainLM/ .
   # OR use cp with exclusions
   cp -r /Users/david/Projects/BrainLM/* .
   cp -r /Users/david/Projects/BrainLM/.* . 2>/dev/null || true
   rm -rf .git
   ```

3. **Verify no participant IDs remain**
   ```bash
   # Search for any remaining IDs
   grep -r "REDACTED_PARTICIPANT_ID" . || echo "✓ REDACTED_PARTICIPANT_ID not found"
   grep -r "REDACTED_PARTICIPANT_ID" . || echo "✓ REDACTED_PARTICIPANT_ID not found"
   grep -r "REDACTED_PARTICIPANT_ID" . || echo "✓ REDACTED_PARTICIPANT_ID not found"
   ```

4. **Initialize new Git repository**
   ```bash
   git init
   git add .
   git commit -m "Initial commit - Clean repository without participant IDs"
   ```

5. **Add remote (pointing to NEW repository on GitHub)**
   ```bash
   # You'll need to create a new repository on GitHub first
   git remote add origin git@github.com:vandijklab/BrainLM.git
   # OR if using a different name temporarily:
   # git remote add origin git@github.com:vandijklab/BrainLM-clean.git
   ```

### Phase 3: Delete Old Repository and Push New One

**Steps**:

1. **Push new clean repository**
   ```bash
   git branch -M main
   git push -u origin main
   ```

2. **Delete old repository on GitHub**
   - Go to GitHub: https://github.com/vandijklab/BrainLM
   - Settings → Scroll to bottom → "Delete this repository"
   - Type repository name to confirm
   - **This is required by UK Biobank**

3. **If using same name, rename new repo OR create with same name**
   - Option A: Delete old repo, then rename new repo to BrainLM
   - Option B: Create new repo with same name after deletion

### Phase 4: Post-Cleanup Actions

1. **Notify collaborators**
   - All team members must delete their local clones
   - Re-clone the repository
   - Any forks must be updated or deleted

2. **Check for forks**
   - Check GitHub for forks of the repository
   - Contact fork owners to remove sensitive data
   - Consider asking GitHub to help if needed

3. **Verify compliance**
   - Run final search: `git log --all -S "REDACTED_PARTICIPANT_ID"` (should return nothing)
   - Search all branches: `git grep -r "REDACTED_PARTICIPANT_ID"` (should return nothing)

## Recommended Approach (Per UK Biobank Requirements)

**This is the REQUIRED approach per UK Biobank:**

1. **Phase 1**: Clear notebook outputs using `jupyter nbconvert` (removes participant IDs)
2. **Phase 2**: Create completely new repository with fresh Git history (no old commits)
3. **Phase 3**: Delete old repository on GitHub, push new clean repository
4. **Phase 4**: Notify team and verify compliance

**Why this approach:**
- ✅ Guarantees no participant data in Git history
- ✅ Simpler than rewriting history
- ✅ Meets UK Biobank's explicit requirement
- ✅ No risk of missing data in old commits

## Important Considerations

⚠️ **WARNINGS**:
- History rewriting is **destructive** and **irreversible** (unless you have backups)
- All collaborators must re-clone the repository
- Any forks will still contain the old history
- GitHub may cache old commits for a period
- This will change all commit hashes

✅ **BEST PRACTICES**:
- Create full backup before starting
- Test on a copy first
- Coordinate with team before force pushing
- Document the cleanup process
- Consider using GitHub's "Private" mode during cleanup

## Important Notes

- **This approach loses all Git commit history** - but this is required by UK Biobank
- The new repository will have a single "Initial commit" 
- All collaborators must delete their local clones and re-clone
- Any forks of the old repository will still contain the old history - you may need to contact fork owners
- GitHub may cache the old repository briefly, but it will be permanently deleted

## Verification Commands

After cleanup, run these to verify:

```bash
# Search all branches and history
git log --all --full-history -S "REDACTED_PARTICIPANT_ID"
git log --all --full-history -S "REDACTED_PARTICIPANT_ID"
git log --all --full-history -S "REDACTED_PARTICIPANT_ID"

# Search current files
git grep -r "REDACTED_PARTICIPANT_ID"
git grep -r "REDACTED_PARTICIPANT_ID"
git grep -r "REDACTED_PARTICIPANT_ID"

# Search notebook outputs specifically
find . -name "*.ipynb" -exec grep -l "REDACTED_PARTICIPANT_ID\|REDACTED_PARTICIPANT_ID\|REDACTED_PARTICIPANT_ID" {} \;
```

All should return empty results.

