import json
from datetime import datetime
now = datetime.now()

meta = {}
meta['title'] = 'Multi-Timescale Power System Dataset'
meta['description'] = '' 
meta['subject'] = 'Power Systems, Dynamic Simulation, Multi-Timescale'
meta['date'] = now.strftime("%d/%m/%Y %H:%M:%S")
meta['publisher'] = 'Texas A&M Univerisity, University of Southern California'
meta['contactPoint'] = 'le.xie@tamu.edu'
meta['creator'] = 'Dongqi Wu, Xiangtian Zheng, Tong Huang, Loc Trinh, Nan Xu, Sivaranjani Seetharaman, Le Xie and Yan Liu'
meta['format'] = '.csv'
meta['type'] = 'open-source power system dataset for machine learning'
meta['contributor'] = 'Dongqi Wu, Xiangtian Zheng, Tong Huang, Loc Trinh, Nan Xu, Sivaranjani Seetharaman, Le Xie and Yan Liu'
meta['identifier'] = 'TBD'
meta['source'] = 'TBD'
meta['language'] = 'Python'
meta['relation'] = 'TBD'
meta['rights'] = 'TBD'

with open('.zenodo.json', 'w') as out:
    json.dump(meta, out, indent=4)
