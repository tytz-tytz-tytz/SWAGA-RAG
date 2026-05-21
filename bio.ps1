New-Item -ItemType Directory -Force -Path data\external

Invoke-WebRequest `
  -Uri "https://ftp.ncbi.nlm.nih.gov/pub/pmc/PMC-ids.csv.gz" `
  -OutFile "data\external\PMC-ids.csv.gz"

python scripts\match_bioasq_pmcs.py --pmc-map data\external\PMC-ids.csv.gz