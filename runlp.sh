for i in (seq 1 5)
  prime-run python -m dvc_personality_fl.main --mode compare --rounds 5 --dataset fsn-mnist >> out_recs/rec_cfr_iid.txt
end
