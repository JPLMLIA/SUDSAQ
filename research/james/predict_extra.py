from glob import glob
from tqdm import tqdm

from sudsaq.data       import flatten, load
from sudsaq.ml.analyze import analyze
from sudsaq.utils      import load_pkl

names = {
    '01': 'jan',
    '02': 'feb',
    '03': 'mar',
    '04': 'apr',
    '05': 'may',
    '06': 'jun',
    '07': 'jul',
    '08': 'aug',
    '09': 'sep',
    '10': 'oct',
    '11': 'nov',
    '12': 'dec'
}

def predict(base):
    """
    """
    years = list(range(2016, 2017+1))
    for year in years:
        for month in tqdm([f'{month:02}' for month in range(1, 13)], desc='Months Processed'):
            config = Config(f"""\
v4:
  plots:
    importances:
      count: 20
      labelrotation: 90
  input:
    sel:
      vars: '[^\.]+\.(?!mda
      8|hno3|oh|pan|q2|sens|so2|T2|taugxs|taugys|taux|tauy|twpc|2dsfc.CFC11|2dsfc.CFC113|2dsfc.CFC12|ch2o|cumf0|2dsfc.dms|2dsfc.HCFC22|2dsfc.H1211|2dsfc.H1301|2dsfc.mc.oc|2dsfc.dflx.bc|2dsfc.dflx.oc|2dsfc.LR.SO2|2dsfc.CH3COOOH|prcpl|2dsfc.C3H7OOH|dqlsc|2dsfc.mc.pm25.salt|2dsfc.CH3COO2|u10|2dsfc.dflx.nh4|2dsfc.mc.nh4|2dsfc.dflx.dust|2dsfc.mc.pm25.dust|osr|osrc|ssrc|v10|2dsfc.OCS|2dsfc.taut|ccoverl|ccoverm|2dsfc.DCDT.HOX|2dsfc.DCDT.OY|2dsfc.DCDT.SO2|slrdc|uvabs|dqcum|dqdad|dqdyn|dqvdf|dtdad|cumf|ccoverh|prcpc|2dsfc.BrCl|2dsfc.Br2|dtcum|2dsfc.mc.sulf|2dsfc.HOBr|dtlsc|2dsfc.Cl2|2dsfc.CH3CCl3|2dsfc.CH3Br|2dsfc.ONMV|2dsfc.MACROOH|2dsfc.MACR|2dsfc.HBr|Restart|agcm|CHEMTMP|SYSIN|gralb|.*gt3|stderr|tcr2).*'
    scale: True
    daily:
      momo:
        vars: momo.(?!mda8).*
        time: [8, 15]
        local: True
patch:
    input:
        glob:
            - /projects/mlia-active-data/data_SUDSAQ/data/momo/{year}/{month}.nc
    output:
        path: {base}/{names[month]}/{year}/
        data         : True
        predict      : True
        bias         : True
        contributions: True
        importance   : True
        plots        : True
            """, 'v4<-patch')
            tqdm.write('Loading')
            data = flatten(
                load(config, lazy=True, split=False)
            ).transpose('loc', 'variable').load()

            models = glob(f'{base}/{names[month]}/**/model.pkl')
            for model in tqdm(models, desc='Models Processed'):
                analyze(
                    model = load_pkl(model),
                    data  = data
                )

predict('/home/jamesmo/sudsaq/models/bias/local/8hr_median/v4/')
