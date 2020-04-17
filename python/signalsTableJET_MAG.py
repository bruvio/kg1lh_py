def signalsTableJET(signalsTableName):

    sigTable = []


    if signalsTableName == 'signalsTable_EFIT':
           sigTable = {
                  # EFIT
                  # 'BPCA': 'PPF/EFIT/BPCA',  # simulated p-poloidal
                  # 'BPME': 'PPF/EFIT/BPME',  # MEASURED p-poloidal
                  # 'FLCA': 'PPF/EFIT/FLCA',  # simulated FLUX AND SADDLE
                  # 'FLME': 'PPF/EFIT/FLME',  # MEASURED FLUX AND SADDLE
                  'RBND': 'PPF/EFIT/RBND',  # r coordinate of boundary
                  'ZBND': 'PPF/EFIT/ZBND',  # z coordinate of boundary
                  # 'PSI':  'PPF/EFIT/PSI',  # 1089x989 psi gse solution  [33x33=1089]
                  # 'PSIR': 'PPF/EFIT/PSIR',  # 33 psi r grid
                  # 'PSIZ': 'PPF/EFIT/PSIZ',  # 33 psi r grid
                  # 'FBND': 'PPF/EFIT/FBND',  # psi at boundary
                  # 'RSIL': 'PPF/EFIT/RSIL',  # R inner lower strike (r of RSIGB)
                  # 'RSIU': 'PPF/EFIT/RSIU',  # R inner upper strike (r of ZSIGB)
                  # 'ZSIL': 'PPF/EFIT/ZSIL',  # Z inner lower strike (z of RSIGB)
                  # 'ZSIU': 'PPF/EFIT/ZSIU',  # Z inner upper strike (z of ZSIGB)
                  # 'RSOL': 'PPF/EFIT/RSOL',  # R outer lower strike (r of RSOGB)
                  # 'RSOU': 'PPF/EFIT/RSOU',  # R outer upper strike (r of ZSOGB)
                  # 'ZSOL': 'PPF/EFIT/ZSOL',  # Z outer lower strike (z of RSOGB)
                  # 'ZSOU': 'PPF/EFIT/ZSOU',  # Z outer upper strike (z of ZSOGB)
                  # 'NBND': 'PPF/EFIT/NBND',  # actual points of RBND,ZBND
                  # 'FAXS': 'PPF/EFIT/FAXS',  # psi at magnetic axis
                  # 'RMAG': 'PPF/EFIT/RMAG',  # r coordinate of magnetic axis
                  # 'ZMAG': 'PPF/EFIT/ZMAG',  # z coordinate of magnetic axis
           }


    elif signalsTableName == 'signalsTable_EHTR':
           sigTable = {
                  # EFIT
                  # 'BPCA': 'PPF/EHTR/BPCA',  # simulated p-poloidal
                  # 'BPME': 'PPF/EHTR/BPME',  # MEASURED p-poloidal
                  # 'FLCA': 'PPF/EHTR/FLCA',  # simulated FLUX AND SADDLE
                  # 'FLME': 'PPF/EHTR/FLME',  # MEASURED FLUX AND SADDLE
                  'RBND': 'PPF/EHTR/RBND',  # r coordinate of boundary
                  'ZBND': 'PPF/EHTR/ZBND',  # z coordinate of boundary
                  # 'PSI':  'PPF/EHTR/PSI',  # 1089x989 psi gse solution  [33x33=1089]
                  # 'PSIR': 'PPF/EHTR/PSIR',  # 33 psi r grid
                  # 'PSIZ': 'PPF/EHTR/PSIZ',  # 33 psi r grid
                  # 'FBND': 'PPF/EHTR/FBND',  # psi at boundary
                  # 'RSIL': 'PPF/EHTR/RSIL',  # R inner lower strike (r of RSIGB)
                  # 'RSIU': 'PPF/EHTR/RSIU',  # R inner upper strike (r of ZSIGB)
                  # 'ZSIL': 'PPF/EHTR/ZSIL',  # Z inner lower strike (z of RSIGB)
                  # 'ZSIU': 'PPF/EHTR/ZSIU',  # Z inner upper strike (z of ZSIGB)
                  # 'RSOL': 'PPF/EHTR/RSOL',  # R outer lower strike (r of RSOGB)
                  # 'RSOU': 'PPF/EHTR/RSOU',  # R outer upper strike (r of ZSOGB)
                  # 'ZSOL': 'PPF/EHTR/ZSOL',  # Z outer lower strike (z of RSOGB)
                  # 'ZSOU': 'PPF/EHTR/ZSOU',  # Z outer upper strike (z of ZSOGB)
                  # 'NBND': 'PPF/EHTR/NBND',  # actual points of RBND,ZBND
                  # 'FAXS': 'PPF/EHTR/FAXS',  # psi at magnetic axis
                  # 'RMAG': 'PPF/EHTR/RMAG',  # r coordinate of magnetic axis
                  # 'ZMAG': 'PPF/EHTR/ZMAG',  # z coordinate of magnetic axis
           }
    elif signalsTableName == 'signalsTable_EFTP':
           sigTable = {
                  # EFIT
                  # 'BPCA': 'PPF/EFTP/BPCA',  # simulated p-poloidal
                  # 'BPME': 'PPF/EFTP/BPME',  # MEASURED p-poloidal
                  # 'FLCA': 'PPF/EFTP/FLCA',  # simulated FLUX AND SADDLE
                  # 'FLME': 'PPF/EFTP/FLME',  # MEASURED FLUX AND SADDLE
                  'RBND': 'PPF/EFTP/RBND',  # r coordinate of boundary
                  'ZBND': 'PPF/EFTP/ZBND',  # z coordinate of boundary
                  # 'PSI': 'PPF/EFTP/PSI',  # 1089x989 psi gse solution  [33x33=1089]
                  # 'PSIR': 'PPF/EFTP/PSIR',  # 33 psi r grid
                  # 'PSIZ': 'PPF/EFTP/PSIZ',  # 33 psi r grid
                  # 'FBND': 'PPF/EFTP/FBND',  # psi at boundary
                  # 'RSIL': 'PPF/EFTP/RSIL',  # R inner lower strike (r of RSIGB)
                  # 'RSIU': 'PPF/EFTP/RSIU',  # R inner upper strike (r of ZSIGB)
                  # 'ZSIL': 'PPF/EFTP/ZSIL',  # Z inner lower strike (z of RSIGB)
                  # 'ZSIU': 'PPF/EFTP/ZSIU',  # Z inner upper strike (z of ZSIGB)
                  # 'RSOL': 'PPF/EFTP/RSOL',  # R outer lower strike (r of RSOGB)
                  # 'RSOU': 'PPF/EFTP/RSOU',  # R outer upper strike (r of ZSOGB)
                  # 'ZSOL': 'PPF/EFTP/ZSOL',  # Z outer lower strike (z of RSOGB)
                  # 'ZSOU': 'PPF/EFTP/ZSOU',  # Z outer upper strike (z of ZSOGB)
                  # 'NBND': 'PPF/EFTP/NBND',  # actual points of RBND,ZBND
                  # 'FAXS': 'PPF/EFTP/FAXS',  # psi at magnetic axis
                  # 'RMAG': 'PPF/EFTP/RMAG',  # r coordinate of magnetic axis
                  # 'ZMAG': 'PPF/EFTP/ZMAG',  # z coordinate of magnetic axis
           }

    return sigTable
