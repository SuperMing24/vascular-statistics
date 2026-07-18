# Bundled manual-cut review font

NotoSansCJKSC-ManualCutSubset.otf is a subset of the Simplified Chinese
face (font index 2) in NotoSansCJK-Regular.ttc from the CentOS/RHEL package
google-noto-sans-cjk-ttc-fonts-20190416-1.el8.noarch.

The subset contains printable ASCII, CJK punctuation, and every non-ASCII
character currently used by manual_cut_review.py. It is loaded by absolute
file path so rendering does not depend on fonts installed on a Slurm compute
node. The font remains licensed under the SIL Open Font License 1.1; see
LICENSE.txt.

Regenerate the subset with FontTools 4.63 or later after adding review text.
Use font index 2 and update the Unicode set to cover the module's non-ASCII
characters.
