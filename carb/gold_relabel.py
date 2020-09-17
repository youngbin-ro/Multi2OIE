from carb.oieReader import OieReader
from carb.extraction import Extraction
from _collections import defaultdict
import json

class Relabel_GoldReader(OieReader):
	
	# Path relative to repo root folder
    default_filename = './oie_corpus/all.oie' 
	
    def __init__(self):
        self.name = 'Relabel_Gold'
	
    def read(self, fn):
        d = defaultdict(lambda: [])
        with open(fn) as fin:
            data = json.load(fin)
        for sentence in data:
            tuples = data[sentence]
            for t in tuples:
                if t["pred"].strip() == "<be>":
                    rel = "[is]"
                else:
                    rel = t["pred"].replace("<be> ","")
                confidence = 1
				
                curExtraction = Extraction(pred = rel,
							               head_pred_index = None,
	                                       sent = sentence,
	                                       confidence = float(confidence),
	                                       index = None)
                if t["arg0"] != "":
                    curExtraction.addArg(t["arg0"])
                if t["arg1"] != "":
                    curExtraction.addArg(t["arg1"])
                if t["arg2"] != "":
                    curExtraction.addArg(t["arg2"])
                if t["arg3"] != "":
                    curExtraction.addArg(t["arg3"])
                if t["temp"] != "":
                    curExtraction.addArg(t["temp"])
                if t["loc"] != "":
                    curExtraction.addArg(t["loc"])
	                
                d[sentence].append(curExtraction)
        self.oie = d