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
    elif signalsTableName == 'signalsTable_XLOC':
        sigTable = {
            # keys  Values
            # XLOC GAPS,STRIKE POINTS and LIMITER/DIVERTED INDICATOR (CTYPE)
            # GAPS
            'LOG': 'PF/SC-LOG<XS',  # 'Lower out gap XLOC'
            'ROG': 'PF/SC-ROG<XS',  # 'Outer gap XLOC'
            'ZUP': 'PF/SC-ZUP<XS',  # 'Top inner gap XLOC'
            'RIG': 'PF/SC-RIG<XS',  # 'Inner gap XLOC'
            # 'SOG'    :  'PF/SC-SOG<XS'       ,     #    'TBD'
            # 'SIG'    :  'PF/SC-SIG<XS'       ,     #    'TBD'
            'TOG1': 'PF/SC-TOG<XS:001',  # 'Top gap XLOC comp 1'
            'TOG2': 'PF/SC-TOG<XS:002',  # 'Top gap XLOC comp 2'
            'TOG3': 'PF/SC-TOG<XS:003',  # 'Top gap XLOC comp 3'
            'TOG4': 'PF/SC-TOG<XS:004',  # 'Top gap XLOC comp 4'
            'TOG5': 'PF/SC-TOG<XS:005',  # 'Top gap XLOC comp 5'
            'GAP2': 'PF/SC-GAP<XS:002',  # 'XLOC gap 2'
            'GAP3': 'PF/SC-GAP<XS:003',  # 'XLOC gap 3'
            'GAP4': 'PF/SC-GAP<XS:004',  # 'XLOC gap 4'
            'GAP6': 'PF/SC-GAP<XS:006',  # 'XLOC gap 6'
            'GAP7': 'PF/SC-GAP<XS:007',  # 'XLOC gap 7'
            'GAP17': 'PF/SC-GAP<XS:017',  # 'XSC gap 1'
            'GAP18': 'PF/SC-GAP<XS:018',  # 'XSC gap 2'
            'GAP19': 'PF/SC-GAP<XS:019',  # 'XSC gap 3'
            'GAP20': 'PF/SC-GAP<XS:020',  # 'XSC gap 4'
            'GAP21': 'PF/SC-GAP<XS:021',  # 'XSC gap 5'
            'GAP22': 'PF/SC-GAP<XS:022',  # 'XSC gap 6'
            'GAP23': 'PF/SC-GAP<XS:023',  # 'XSC gap 7'
            'GAP24': 'PF/SC-GAP<XS:024',  # 'XSC gap 8'
            'GAP25': 'PF/SC-GAP<XS:025',  # 'XSC gap 9'
            'GAP26': 'PF/SC-GAP<XS:026',  # 'XSC gap 10'
            'GAP27': 'PF/SC-GAP<XS:027',  # 'XSC gap 11'
            'GAP28': 'PF/SC-GAP<XS:028',  # 'XSC gap 12'
            'GAP29': 'PF/SC-GAP<XS:029',  # 'XSC gap 13'
            'GAP30': 'PF/SC-GAP<XS:030',  # 'XSC gap 14'
            'GAP31': 'PF/SC-GAP<XS:031',  # 'XSC gap 15'
            'GAP32': 'PF/SC-GAP<XS:032',  # 'XSC gap 16'
            #  STRIKE POINTS
            'RSOGB': 'PF/SC-RSOGB<XS',  # 'In-tgt st-pt XLOC'
            'RSIGB': 'PF/SC-RSIGB<XS',  # 'In-side st-pt XLOC'
            'ZSOGB': 'PF/SC-ZSOGB<XS',  # 'Out-tgt st-pt XLOC'
            'ZSIGB': 'PF/SC-ZSIGB<XS',  # 'Out-side st-pt XLOC'
            'WLBSRP': 'PF/SC-WLBSRP<XS',  # '
            # X POINT
            'RX': 'PF/SC-RXGB<XS',  # 'X-point radius XLOC'
            'ZX': 'PF/SC-ZXGB<XS',  # 'X-point height XLOC'
            'CTYPE': 'PF/SC-CTYPE<XS',  # 'XLOC config type: -1 is xpoint'
        }
    return sigTable
