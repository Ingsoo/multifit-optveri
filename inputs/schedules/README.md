Place schedule-plot input files here.

Recommended pattern:

- one instance per file
- use `.txt`
- separate processing times with commas, spaces, semicolons, or one per line

Example:

```text
9/17
7/17
6/17
5/17
5/17
```

Then run:

```powershell
python scripts/plot_schedules.py --machines 8 --instance my_case
```

for `inputs/schedules/my_case.txt`, or:

```powershell
python scripts/plot_schedules.py --machines 8 --jobs-file my_case.txt
```
