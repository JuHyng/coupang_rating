cd ../../Coupang-Review-Crawling

export CATEGORY="식품" 
export URL_FILE="/home/dl/TP_DL/AL2/Coupang-Review-Crawling/test_식품.txt" 
export PAGE_COUNT="5" 

python main.py \
    $CATEGORY \
    $URL_FILE \
    $PAGE_COUNT