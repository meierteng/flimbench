#!/bin/bash
#SBATCH --job-name=sb-dl
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=06:00:00
#SBATCH --output=/export/fs01/cli229/style_bench/logs/%x-%j.out
# Download the six-director CMD clips (video only, best stream up to 1080p) into fs01.
B=/export/fs01/cli229/style_bench
cd $B
cut -d, -f1 clip_list_6dir.csv | tail -n +2 | sed 's#^#https://www.youtube.com/watch?v=#' > urls.txt
$B/venv/bin/yt-dlp -a urls.txt -f "bv*[height<=1080][vcodec^=vp09]/bv*[height<=1080][vcodec^=vp9]/bv*[height<=1080][vcodec^=avc1]" -o "data/clips/%(id)s.%(ext)s" \
  --download-archive data/archive.txt --no-playlist --ignore-errors --no-warnings --no-progress \
  --sleep-requests 1 --sleep-interval 2 --max-sleep-interval 6 \
  --print-to-file "after_move:%(id)s,%(height)s,%(width)s,%(vcodec)s,%(fps)s,%(tbr)s" data/clip_formats.csv \
  2> logs/dl_errors.txt
echo done; ls data/clips | wc -l; du -sh data/clips
