def signalsTableJET(signalsTableName):

    sigTable = []

    if signalsTableName == 'signalsTable_XLOC_KC1D':
           sigTable = {
           # keys  Values
           # KC1D JPF IDC OCT.3&7 pick up coils
           'CZ01' :'DA/C2-CX01',   # oct. 3 IDC
           'CZ02' :'DA/C2-CX02',   # oct. 3 IDC
           'CZ03' :'DA/C2-CX03',   # oct. 3 IDC
           'CZ04' :'DA/C2-CX04',   # oct. 3 IDC
           'CZ05' :'DA/C2-CX05',   # oct. 3 IDC
           'CZ06' :'DA/C2-CX06',   # oct. 3 IDC
           'CZ07' :'DA/C2-CX07',   # oct. 3 IDC
           'CZ08' :'DA/C2-CX08',   # oct. 3 IDC
           'CZ09' :'DA/C2-CX09',   # oct. 3 IDC
           'CZ10' :'DA/C2-CX10',   # oct. 3 IDC
           'CZ11' :'DA/C2-CY11',   # oct. 7 IDC
           'CZ12' :'DA/C2-CX12',   # oct. 3 IDC
           'CZ13' :'DA/C2-CX13',   # oct. 3 IDC
           'CZ14' :'DA/C2-CX14',   # oct. 3 IDC
           'CZ15' :'DA/C2-CX15',   # oct. 3 IDC
           'CZ16' :'DA/C2-CX16',   # oct. 3 IDC
           'CZ17' :'DA/C2-CX17',   # oct. 3 IDC
           'CZ18' :'DA/C2-CY18',   # oct. 7 IDC
           # KC1D JPF IDC OCT.3 saddle loops
           'SZ01' :'DA/C2-SX01',   # oct. 3 Saddle Loops
           'SZ02' :'DA/C2-SX02',   # oct. 3 Saddle Loops
           'SZ03' :'DA/C2-SX03',   # oct. 3 Saddle Loops
           'SZ04' :'DA/C2-SY04',   # oct. 7 Saddle Loops
           'SZ05' :'DA/C2-SY05',   # oct. 7 Saddle Loops
           'SZ06' :'DA/C2-SX06',   # oct. 3 Saddle Loops
           'SZ07' :'DA/C2-SX07',   # oct. 3 Saddle Loops
           'SZ08' :'DA/C2-SX08',   # oct. 3 Saddle Loops
           'SZ09' :'DA/C2-SX09',   # oct. 3 Saddle Loops
           'SZ10' :'DA/C2-SX10',   # oct. 3 Saddle Loops
           'SZ11' :'DA/C2-SX11',   # oct. 3 Saddle Loops
           'SZ12' :'DA/C2-SX12',   # oct. 3 Saddle Loops
           'SZ13' :'DA/C2-SX13',   # oct. 3 Saddle Loops
           'SZ14' :'DA/C2-SX14',   # oct. 3 Saddle Loops
           # KC1D JPF full flux loops
           'FLD2' :'DA/C2-FLD2',   # full flux loop
           'FLD3' :'DA/C2-FLD3',   # full flux loop
           'FL05' :'DA/C2-FL05',   # full flux loop
           'FL11' :'DA/C2-FL11',   # full flux loop
           # KC1D divertor Probes 1996 tangential
           'TPZ01' :'DA/C2-TP601',   # oct 7 tangential mod.19
           'TPZ02' :'DA/C2-TP602',   # oct 7 tangential mod.19
           'TPZ03' :'DA/C2-TP603',   # oct 7 tangential mod.19
           'TPZ04' :'DA/C2-TP604',   # oct 7 tangential mod.19
           #'TPZ05' :'DA/C2-TP605',   # oct 7 tangential mod.19
           'TPZ06' :'DA/C2-TP606',   # oct 7 tangential mod.19
           #'TPZ07' :'DA/C2-TP607',   # oct 7 tangential mod.19
           'TPZ08' :'DA/C2-TP608',   # oct 7 tangential mod.19
           #'TPZ09' :'DA/C2-TP609',   # oct 7 tangential mod.19
           #'TPZ10' :'DA/C2-TP610',   # oct 7 tangential mod.19
           'TPZ11' :'DA/C2-TP611',   # oct 7 tangential mod.19
           # KC1D divertor Probes 1996 tangential
           'TNZ01': 'DA/C2-TN601',  # oct 7 tangential mod.19
           'TNZ02': 'DA/C2-TN602',  # oct 7 tangential mod.19
           'TNZ03': 'DA/C2-TN603',  # oct 7 tangential mod.19
           'TNZ04': 'DA/C2-TN604',  # oct 7 tangential mod.19
           #'TNZ05' :'DA/C2-TNZ05',   # oct 7 tangential mod.19
           'TNZ06': 'DA/C2-TN206',  # oct 3 tangential mod.7
           #'TNZ07' :'DA/C2-TNZ07',   # oct 7 tangential mod.19
           'TNZ08': 'DA/C2-TN608',  # oct 7 tangential mod.19
           'TNZ09' :'DA/C2-TN609',   # oct 7 tangential mod.19
           'TNZ10' :'DA/C2-TN610',   # oct 7 tangential mod.19
           'TNZ11': 'DA/C2-TN611',  # oct 7 tangential mod.19
           # KC1D Poloidal Limiter coils
           'PZ02': 'DA/C2-P802',  # oct 8
           'PZ03': 'DA/C2-P803',  # oct 8
           'PZ04': 'DA/C2-P804',  # oct 8
           'PZ05': 'DA/C2-P805',  # oct 8
           # divertor current
           'ID1'      :'DA/C2-ID1',  # D1 current received by SC
           'ID2'      :'DA/C2-ID2',  # D2 current received by SC
           'ID3'      :'DA/C2-ID3',  # D3 current received by SC
           'ID4'      :'DA/C2-ID4',  # D4 current received by SC
            }
    if signalsTableName == 'signalsTable_XLOC_KC1Z':
           sigTable = {
           # keys  Values
           # KC1Z JPF IDC OCT.3&7 pick up coils
           'CZ01' :'DA/C2Z-CX01',   # oct. 3 IDC
           'CZ02' :'DA/C2Z-CX02',   # oct. 3 IDC
           'CZ03' :'DA/C2Z-CX03',   # oct. 3 IDC
           'CZ04' :'DA/C2Z-CX04',   # oct. 3 IDC
           'CZ05' :'DA/C2Z-CX05',   # oct. 3 IDC
           'CZ06' :'DA/C2Z-CX06',   # oct. 3 IDC
           'CZ07' :'DA/C2Z-CX07',   # oct. 3 IDC
           'CZ08' :'DA/C2Z-CX08',   # oct. 3 IDC
           'CZ09' :'DA/C2Z-CX09',   # oct. 3 IDC
           'CZ10' :'DA/C2Z-CX10',   # oct. 3 IDC
           'CZ11' :'DA/C2Z-CY11',   # oct. 7 IDC
           'CZ12' :'DA/C2Z-CX12',   # oct. 3 IDC
           'CZ13' :'DA/C2Z-CX13',   # oct. 3 IDC
           'CZ14' :'DA/C2Z-CX14',   # oct. 3 IDC
           'CZ15' :'DA/C2Z-CX15',   # oct. 3 IDC
           'CZ16' :'DA/C2Z-CX16',   # oct. 3 IDC
           'CZ17' :'DA/C2Z-CX17',   # oct. 3 IDC
           'CZ18' :'DA/C2Z-CY18',   # oct. 7 IDC
           # KC1Z JPF IDC OCT.3 saddle loops
           'SZ01' :'DA/C2Z-SX01',   # oct. 3 Saddle Loops
           'SZ02' :'DA/C2Z-SX02',   # oct. 3 Saddle Loops
           'SZ03' :'DA/C2Z-SX03',   # oct. 3 Saddle Loops
           'SZ04' :'DA/C2Z-SY04',   # oct. 7 Saddle Loops
           'SZ05' :'DA/C2Z-SY05',   # oct. 7 Saddle Loops
           'SZ06' :'DA/C2Z-SX06',   # oct. 3 Saddle Loops
           'SZ07' :'DA/C2Z-SX07',   # oct. 3 Saddle Loops
           'SZ08' :'DA/C2Z-SX08',   # oct. 3 Saddle Loops
           'SZ09' :'DA/C2Z-SX09',   # oct. 3 Saddle Loops
           'SZ10' :'DA/C2Z-SX10',   # oct. 3 Saddle Loops
           'SZ11' :'DA/C2Z-SX11',   # oct. 3 Saddle Loops
           'SZ12' :'DA/C2Z-SX12',   # oct. 3 Saddle Loops
           'SZ13' :'DA/C2Z-SX13',   # oct. 3 Saddle Loops
           'SZ14' :'DA/C2Z-SX14',   # oct. 3 Saddle Loops
           # KC1Z JPF full flux loops
           'FLD2' :'DA/C2Z-FLD2',   # full flux loop
           'FLD3' :'DA/C2Z-FLD3',   # full flux loop
           'FL05' :'DA/C2Z-FL05',   # full flux loop
           'FL11' :'DA/C2Z-FL11',   # full flux loop
           # KC1Z divertor Probes 1996 tangential
           'TPZ01' :'DA/C2Z-TP601',   # oct 7 tangential mod.19
           'TPZ02' :'DA/C2Z-TP602',   # oct 7 tangential mod.19
           'TPZ03' :'DA/C2Z-TP603',   # oct 7 tangential mod.19
           'TPZ04' :'DA/C2Z-TP604',   # oct 7 tangential mod.19
           #'TPZ05' :'DA/C2Z-TP605',   # oct 7 tangential mod.19
           'TPZ06' :'DA/C2Z-TP606',   # oct 7 tangential mod.19
           #'TPZ07' :'DA/C2Z-TP607',   # oct 7 tangential mod.19
           'TPZ08' :'DA/C2Z-TP608',   # oct 7 tangential mod.19
           #'TPZ09' :'DA/C2Z-TP609',   # oct 7 tangential mod.19
           #'TPZ10' :'DA/C2Z-TP610',   # oct 7 tangential mod.19
           'TPZ11' :'DA/C2Z-TP611',   # oct 7 tangential mod.19
           # KC1Z divertor Probes 1996 tangential
           'TNZ01': 'DA/C2Z-TN601',  # oct 7 tangential mod.19
           'TNZ02': 'DA/C2Z-TN602',  # oct 7 tangential mod.19
           'TNZ03': 'DA/C2Z-TN603',  # oct 7 tangential mod.19
           'TNZ04': 'DA/C2Z-TN604',  # oct 7 tangential mod.19
           #'TNZ05' :'DA/C2Z-TNZ05',   # oct 7 tangential mod.19
           'TNZ06': 'DA/C2Z-TN206',  # oct 3 tangential mod.7
           #'TNZ07' :'DA/C2Z-TNZ07',   # oct 7 tangential mod.19
           'TNZ08': 'DA/C2Z-TN608',  # oct 7 tangential mod.19
           'TNZ09' :'DA/C2Z-TN609',   # oct 7 tangential mod.19
           'TNZ10' :'DA/C2Z-TN610',   # oct 7 tangential mod.19
           'TNZ11': 'DA/C2Z-TN611',  # oct 7 tangential mod.19
           # KC1Z Poloidal Limiter coils
           'PZ02': 'DA/C2Z-P802',  # oct 8
           'PZ03': 'DA/C2Z-P803',  # oct 8
           'PZ04': 'DA/C2Z-P804',  # oct 8
           'PZ05': 'DA/C2Z-P805',  # oct 8
           # divertor current
           'ID1'      :'DA/C2Z-ID1',  # D1 current received by SC
           'ID2'      :'DA/C2Z-ID2',  # D2 current received by SC
           'ID3'      :'DA/C2Z-ID3',  # D3 current received by SC
           'ID4'      :'DA/C2Z-ID4',  # D4 current received by SC
            }
    if signalsTableName == 'signalsTable_KC1D_PU':
           sigTable = {
           # keys  Values
           # KC1D JPF IDC OCT.3 pick up coils
           'CX01' :'DA/C2-CX01',   # oct. 3 IDC
           'CX02' :'DA/C2-CX02',   # oct. 3 IDC
           'CX03' :'DA/C2-CX03',   # oct. 3 IDC
           'CX04' :'DA/C2-CX04',   # oct. 3 IDC
           'CX05' :'DA/C2-CX05',   # oct. 3 IDC
           'CX06' :'DA/C2-CX06',   # oct. 3 IDC
           'CX07' :'DA/C2-CX07',   # oct. 3 IDC
           'CX08' :'DA/C2-CX08',   # oct. 3 IDC
           'CX09' :'DA/C2-CX09',   # oct. 3 IDC
           'CX10' :'DA/C2-CX10',   # oct. 3 IDC
           'CX11' :'DA/C2-CX11',   # oct. 3 IDC
           'CX12' :'DA/C2-CX12',   # oct. 3 IDC
           'CX13' :'DA/C2-CX13',   # oct. 3 IDC
           'CX14' :'DA/C2-CX14',   # oct. 3 IDC
           'CX15' :'DA/C2-CX15',   # oct. 3 IDC
           'CX16' :'DA/C2-CX16',   # oct. 3 IDC
           'CX17' :'DA/C2-CX17',   # oct. 3 IDC
           'CX18' :'DA/C2-CX18',   # oct. 3 IDC
           #
           # keys  Values
           # KC1D JPF IDC OCT.7 pick up coils
           'CY01' :'DA/C2-CY01',   # oct. 7 IDC
           'CY02' :'DA/C2-CY02',   # oct. 7 IDC
           'CY03' :'DA/C2-CY03',   # oct. 7 IDC
           'CY04' :'DA/C2-CY04',   # oct. 7 IDC
           'CY05' :'DA/C2-CY05',   # oct. 7 IDC
           'CY06' :'DA/C2-CY06',   # oct. 7 IDC
           'CY07' :'DA/C2-CY07',   # oct. 7 IDC
           'CY08' :'DA/C2-CY08',   # oct. 7 IDC
           'CY09' :'DA/C2-CY09',   # oct. 7 IDC
           'CY10' :'DA/C2-CY10',   # oct. 7 IDC
           'CY11' :'DA/C2-CY11',   # oct. 7 IDC
           'CY12' :'DA/C2-CY12',   # oct. 7 IDC
           'CY13' :'DA/C2-CY13',   # oct. 7 IDC
           'CY14' :'DA/C2-CY14',   # oct. 7 IDC
           'CY15' :'DA/C2-CY15',   # oct. 7 IDC
           'CY16' :'DA/C2-CY16',   # oct. 7 IDC
           'CY17' :'DA/C2-CY17',   # oct. 7 IDC
           'CY18' :'DA/C2-CY18',   # oct. 7 IDC
           }
    elif signalsTableName == 'signalsTable_KC1Z_PU':
           sigTable = {
           # keys  Values
           # KC1Z JPF IDC OCT.3 pick up coils
           'CX01' :'DA/C2Z-CX01',   # oct. 3 IDC
           'CX02' :'DA/C2Z-CX02',   # oct. 3 IDC
           'CX03' :'DA/C2Z-CX03',   # oct. 3 IDC
           'CX04' :'DA/C2Z-CX04',   # oct. 3 IDC
           'CX05' :'DA/C2Z-CX05',   # oct. 3 IDC
           'CX06' :'DA/C2Z-CX06',   # oct. 3 IDC
           'CX07' :'DA/C2Z-CX07',   # oct. 3 IDC
           'CX08' :'DA/C2Z-CX08',   # oct. 3 IDC
           'CX09' :'DA/C2Z-CX09',   # oct. 3 IDC
           'CX10' :'DA/C2Z-CX10',   # oct. 3 IDC
           'CX11' :'DA/C2Z-CX11',   # oct. 3 IDC
           'CX12' :'DA/C2Z-CX12',   # oct. 3 IDC
           'CX13' :'DA/C2Z-CX13',   # oct. 3 IDC
           'CX14' :'DA/C2Z-CX14',   # oct. 3 IDC
           'CX15' :'DA/C2Z-CX15',   # oct. 3 IDC
           'CX16' :'DA/C2Z-CX16',   # oct. 3 IDC
           'CX17' :'DA/C2Z-CX17',   # oct. 3 IDC
           'CX18' :'DA/C2Z-CX18',   # oct. 3 IDC
           #
           # keys  Values
           # KC1Z JPF IDC OCT.7 pick up coils
           'CY01' :'DA/C2Z-CY01',   # oct. 7 IDC
           'CY02' :'DA/C2Z-CY02',   # oct. 7 IDC
           'CY03' :'DA/C2Z-CY03',   # oct. 7 IDC
           'CY04' :'DA/C2Z-CY04',   # oct. 7 IDC
           'CY05' :'DA/C2Z-CY05',   # oct. 7 IDC
           'CY06' :'DA/C2Z-CY06',   # oct. 7 IDC
           'CY07' :'DA/C2Z-CY07',   # oct. 7 IDC
           'CY08' :'DA/C2Z-CY08',   # oct. 7 IDC
           'CY09' :'DA/C2Z-CY09',   # oct. 7 IDC
           'CY10' :'DA/C2Z-CY10',   # oct. 7 IDC
           'CY11' :'DA/C2Z-CY11',   # oct. 7 IDC
           'CY12' :'DA/C2Z-CY12',   # oct. 7 IDC
           'CY13' :'DA/C2Z-CY13',   # oct. 7 IDC
           'CY14' :'DA/C2Z-CY14',   # oct. 7 IDC
           'CY15' :'DA/C2Z-CY15',   # oct. 7 IDC
           'CY16' :'DA/C2Z-CY16',   # oct. 7 IDC
           'CY17' :'DA/C2Z-CY17',   # oct. 7 IDC
           'CY18' :'DA/C2Z-CY18',   # oct. 7 IDC
           }
    elif signalsTableName == 'signalsTable_KC1D_SL':
           sigTable = {
           # keys  Values
           # KC1D JPF OCT.3 saddle loops
           'SX01' :'DA/C2-SX01',   # oct. 3 Saddle Loops
           'SX02' :'DA/C2-SX02',   # oct. 3 Saddle Loops
           'SX03' :'DA/C2-SX03',   # oct. 3 Saddle Loops
           'SX04' :'DA/C2-SX04',   # oct. 3 Saddle Loops
           'SX05' :'DA/C2-SX05',   # oct. 3 Saddle Loops
           'SX06' :'DA/C2-SX06',   # oct. 3 Saddle Loops
           'SX07' :'DA/C2-SX07',   # oct. 3 Saddle Loops
           'SX08' :'DA/C2-SX08',   # oct. 3 Saddle Loops
           'SX09' :'DA/C2-SX09',   # oct. 3 Saddle Loops
           'SX10' :'DA/C2-SX10',   # oct. 3 Saddle Loops
           'SX11' :'DA/C2-SX11',   # oct. 3 Saddle Loops
           'SX12' :'DA/C2-SX12',   # oct. 3 Saddle Loops
           'SX13' :'DA/C2-SX13',   # oct. 3 Saddle Loops
           'SX14' :'DA/C2-SX14',   # oct. 3 Saddle Loops
           #
           # keys  Values
           # KC1D JPF OCT.7 saddle loops
           'SY01' :'DA/C2-SY01',   # oct. 7 Saddle Loops
           'SY02' :'DA/C2-SY02',   # oct. 7 Saddle Loops
           'SY03' :'DA/C2-SY03',   # oct. 7 Saddle Loops
           'SY04' :'DA/C2-SY04',   # oct. 7 Saddle Loops
           'SY05' :'DA/C2-SY05',   # oct. 7 Saddle Loops
           'SY06' :'DA/C2-SY06',   # oct. 7 Saddle Loops
           'SY07' :'DA/C2-SY07',   # oct. 7 Saddle Loops
           'SY08' :'DA/C2-SY08',   # oct. 7 Saddle Loops
           'SY09' :'DA/C2-SY09',   # oct. 7 Saddle Loops
           'SY10' :'DA/C2-SY10',   # oct. 7 Saddle Loops
           'SY11' :'DA/C2-SY11',   # oct. 7 Saddle Loops
           'SY12' :'DA/C2-SY12',   # oct. 7 Saddle Loops
           'SY13' :'DA/C2-SY13',   # oct. 7 Saddle Loops
           'SY14' :'DA/C2-SY14',   # oct. 7 Saddle Loops
           }
    elif signalsTableName == 'signalsTable_KC1Z_SL':
           sigTable = {
           # keys  Values
           # KC1Z JPF OCT.3 saddle loops
           'SX01' :'DA/C2Z-SX01',   # oct. 3 Saddle Loops
           'SX02' :'DA/C2Z-SX02',   # oct. 3 Saddle Loops
           'SX03' :'DA/C2Z-SX03',   # oct. 3 Saddle Loops
           'SX04' :'DA/C2Z-SX04',   # oct. 3 Saddle Loops
           'SX05' :'DA/C2Z-SX05',   # oct. 3 Saddle Loops
           'SX06' :'DA/C2Z-SX06',   # oct. 3 Saddle Loops
           'SX07' :'DA/C2Z-SX07',   # oct. 3 Saddle Loops
           'SX08' :'DA/C2Z-SX08',   # oct. 3 Saddle Loops
           'SX09' :'DA/C2Z-SX09',   # oct. 3 Saddle Loops
           'SX10' :'DA/C2Z-SX10',   # oct. 3 Saddle Loops
           'SX11' :'DA/C2Z-SX11',   # oct. 3 Saddle Loops
           'SX12' :'DA/C2Z-SX12',   # oct. 3 Saddle Loops
           'SX13' :'DA/C2Z-SX13',   # oct. 3 Saddle Loops
           'SX14' :'DA/C2Z-SX14',   # oct. 3 Saddle Loops
           #
           # keys  Values
           # KC1Z JPF OCT.7 saddle loops
           'SY01' :'DA/C2Z-SY01',   # oct. 7 Saddle Loops
           'SY02' :'DA/C2Z-SY02',   # oct. 7 Saddle Loops
           'SY03' :'DA/C2Z-SY03',   # oct. 7 Saddle Loops
           'SY04' :'DA/C2Z-SY04',   # oct. 7 Saddle Loops
           'SY05' :'DA/C2Z-SY05',   # oct. 7 Saddle Loops
           'SY06' :'DA/C2Z-SY06',   # oct. 7 Saddle Loops
           'SY07' :'DA/C2Z-SY07',   # oct. 7 Saddle Loops
           'SY08' :'DA/C2Z-SY08',   # oct. 7 Saddle Loops
           'SY09' :'DA/C2Z-SY09',   # oct. 7 Saddle Loops
           'SY10' :'DA/C2Z-SY10',   # oct. 7 Saddle Loops
           'SY11' :'DA/C2Z-SY11',   # oct. 7 Saddle Loops
           'SY12' :'DA/C2Z-SY12',   # oct. 7 Saddle Loops
           'SY13' :'DA/C2Z-SY13',   # oct. 7 Saddle Loops
           'SY14' :'DA/C2Z-SY14',   # oct. 7 Saddle Loops
           }
    elif signalsTableName == 'signalsTable_ID_VLrr_PPCC':
           sigTable = {
           # keys  Values
           # KC1D JPF ID and Lotage loops Restraint Rings
           'ID1'      :'DA/C2-ID1',  # D1 current received by SC
           #'ID1_SC'   :'PF/SC-ID1<MS', # SC version of the previous one
           #'ID1_KC1D' :'PF/SC-ID1<KS', # KC1D sends via ATM to SC
           #
           'ID2'      :'DA/C2-ID2',  # D2 current received by SC
           #'ID2_SC'   :'PF/SC-ID2<MS', # SC version of the previous one
           #'ID2_KC1D' :'PF/SC-ID2<KS', # KC1D sends via ATM to SC
           #
           'ID3'      :'DA/C2-ID3',  # D3 current received by SC
           #'ID3_SC'   :'PF/SC-ID3<MS', # SC version of the previous one
           #'ID3_KC1D' :'PF/SC-ID3<KS', # KC1D sends via ATM to SC
           #
           'ID4'      :'DA/C2-ID4',  # D4 current received by SC
           #'ID4_SC'   :'PF/SC-ID4<MS', # SC version of the previous one
           #'ID4_KC1D' :'PF/SC-ID4<KS', # KC1D sends via ATM to SC
           #
           # voltage loop of restaint rings
           'VLRRU'    :'DA/C2-VLRRU',
           'VLRRL'    :'DA/C2-VLRRL',
           #
           'IPLA_SC_KC1E'      :'PF/SC-IPLA<KE',  # IPLA computed using KC1E
           'IPLA_SC_KC1D'      :'PF/SC-IPLA<KS',  # IPLA computed using KC1D
           'IPLA_C2'           :'DA/C2-IPLA',
           'IPLA_C2Z'          :'DA/C2Z-IPLA>',
           'IPLA_CDE'          :'DA/CDE-IPLA',
           'IPLA_CF'           :'DA/CF-IPLA',
           # PPF
           'IPLA_MAGF'         :'PPF/MAGF/IPLA',
           'IPLA_MAGJ'         :'PPF/MAGJ/IPLA',
           'IPLA_MAGN'         :'PPF/MAGN/IPLA',
           #'IPLA_MGZF'         :'PPF/MGZF/IPLA',
           #'IPLA_MGZJ'         :'PPF/MGZJ/IPLA'
           }
    elif signalsTableName == 'signalsTable_KC1E_PU':
           sigTable = {
           # KC1E JPF IDC OCT.1 pick up coils
           'C101' :'DA/C2E-C101',   # oct. 1 IDC
           'C102' :'DA/C2E-C102',   # oct. 1 IDC
           'C103' :'DA/C2E-C103',   # oct. 1 IDC
           'C104' :'DA/C2E-C104',   # oct. 1 IDC
           'C105' :'DA/C2E-C105',   # oct. 1 IDC
           'C106' :'DA/C2E-C106',   # oct. 1 IDC
           'C107' :'DA/C2E-C107',   # oct. 1 IDC
           'C108' :'DA/C2E-C108',   # oct. 1 IDC
           'C109' :'DA/C2E-C109',   # oct. 1 IDC
           'C110' :'DA/C2E-C110',   # oct. 1 IDC
           'C111' :'DA/C2E-C111',   # oct. 1 IDC
           'C112' :'DA/C2E-C112',   # oct. 1 IDC
           'C113' :'DA/C2E-C113',   # oct. 1 IDC
           'C114' :'DA/C2E-C114',   # oct. 1 IDC
           'C115' :'DA/C2E-C115',   # oct. 1 IDC
           'C116' :'DA/C2E-C116',   # oct. 1 IDC
           'C117' :'DA/C2E-C117',   # oct. 1 IDC
           'C118' :'DA/C2E-C118',   # oct. 1 IDC
           #
           # KC1E JPF IDC OCT.5 pick up coils
           'C501' :'DA/C2E-C501',   # oct. 5 IDC
           'C502' :'DA/C2E-C502',   # oct. 5 IDC
           'C503' :'DA/C2E-C503',   # oct. 5 IDC
           'C504' :'DA/C2E-C504',   # oct. 5 IDC
           'C505' :'DA/C2E-C505',   # oct. 5 IDC
           'C506' :'DA/C2E-C506',   # oct. 5 IDC
           'C507' :'DA/C2E-C507',   # oct. 5 IDC
           'C508' :'DA/C2E-C508',   # oct. 5 IDC
           'C509' :'DA/C2E-C509',   # oct. 5 IDC
           'C510' :'DA/C2E-C510',   # oct. 5 IDC
           'C511' :'DA/C2E-C511',   # oct. 5 IDC
           'C512' :'DA/C2E-C512',   # oct. 5 IDC
           'C513' :'DA/C2E-C513',   # oct. 5 IDC
           'C514' :'DA/C2E-C514',   # oct. 5 IDC
           'C515' :'DA/C2E-C515',   # oct. 5 IDC
           'C516' :'DA/C2E-C516',   # oct. 5 IDC
           'C517' :'DA/C2E-C517',   # oct. 5 IDC
           'C518' :'DA/C2E-C518',   # oct. 5 IDC
           }
    elif signalsTableName == 'signalsTable_KC1E_SL':
           sigTable = {
           # KC1E JPF OCT.1 saddle loops
           'S101' :'DA/C2E-S101',   # oct. 1 Saddle Loops
           'S102' :'DA/C2E-S102',   # oct. 1 Saddle Loops
           'S103' :'DA/C2E-S103',   # oct. 1 Saddle Loops
           'S104' :'DA/C2E-S104',   # oct. 1 Saddle Loops
           'S105' :'DA/C2E-S105',   # oct. 1 Saddle Loops
           'S106' :'DA/C2E-S106',   # oct. 1 Saddle Loops
           'S107' :'DA/C2E-S107',   # oct. 1 Saddle Loops
           'S108' :'DA/C2E-S108',   # oct. 1 Saddle Loops
           'S109' :'DA/C2E-S109',   # oct. 1 Saddle Loops
           'S110' :'DA/C2E-S110',   # oct. 1 Saddle Loops
           'S111' :'DA/C2E-S111',   # oct. 1 Saddle Loops
           'S112' :'DA/C2E-S112',   # oct. 1 Saddle Loops
           'S113' :'DA/C2E-S113',   # oct. 1 Saddle Loops
           'S114' :'DA/C2E-S114',   # oct. 1 Saddle Loops
           #
           # KC1E JPF OCT.5 saddle loops
           'S501' :'DA/C2E-S501',   # oct. 5 Saddle Loops
           'S502' :'DA/C2E-S502',   # oct. 5 Saddle Loops
           'S503' :'DA/C2E-S503',   # oct. 5 Saddle Loops
           'S504' :'DA/C2E-S504',   # oct. 5 Saddle Loops
           'S505' :'DA/C2E-S505',   # oct. 5 Saddle Loops
           'S506' :'DA/C2E-S506',   # oct. 5 Saddle Loops
           'S507' :'DA/C2E-S507',   # oct. 5 Saddle Loops
           'S508' :'DA/C2E-S508',   # oct. 5 Saddle Loops
           'S509' :'DA/C2E-S509',   # oct. 5 Saddle Loops
           'S510' :'DA/C2E-S510',   # oct. 5 Saddle Loops
           'S511' :'DA/C2E-S511',   # oct. 5 Saddle Loops
           'S512' :'DA/C2E-S512',   # oct. 5 Saddle Loops
           'S513' :'DA/C2E-S513',   # oct. 5 Saddle Loops
           'S514' :'DA/C2E-S514',   # oct. 5 Saddle Loops
           }
    elif signalsTableName == 'signalsTable_SC_KS_PU':
           sigTable = {
           # Shape Controller (SC) JPF from KC1D IDC
           'CZ01' :'PF/SC-CZ01<KS',   # SC IDC
           'CZ02' :'PF/SC-CZ02<KS',   # SC IDC
           'CZ03' :'PF/SC-CZ03<KS',   # SC IDC
           'CZ04' :'PF/SC-CZ04<KS',   # SC IDC
           'CZ05' :'PF/SC-CZ05<KS',   # SC IDC
           'CZ06' :'PF/SC-CZ06<KS',   # SC IDC
           'CZ07' :'PF/SC-CZ07<KS',   # SC IDC
           'CZ08' :'PF/SC-CZ08<KS',   # SC IDC
           'CZ09' :'PF/SC-CZ09<KS',   # SC IDC
           'CZ10' :'PF/SC-CZ10<KS',   # SC IDC
           'CZ11' :'PF/SC-CZ11<KS',   # SC IDC
           'CZ12' :'PF/SC-CZ12<KS',   # SC IDC
           'CZ13' :'PF/SC-CZ13<KS',   # SC IDC
           'CZ14' :'PF/SC-CZ14<KS',   # SC IDC
           'CZ15' :'PF/SC-CZ15<KS',   # SC IDC
           'CZ16' :'PF/SC-CZ16<KS',   # SC IDC
           'CZ17' :'PF/SC-CZ17<KS',   # SC IDC
           'CZ18' :'PF/SC-CZ18<KS',   # SC IDC
           }
    elif signalsTableName == 'signalsTable_SC_KE_PU':
           sigTable = {
           # Shape Controller (SC) JPF from KC1D IDC
           'CZ01' :'PF/SC-CZ01<KE',   # SC IDC
           'CZ02' :'PF/SC-CZ02<KE',   # SC IDC
           'CZ03' :'PF/SC-CZ03<KE',   # SC IDC
           'CZ04' :'PF/SC-CZ04<KE',   # SC IDC
           'CZ05' :'PF/SC-CZ05<KE',   # SC IDC
           'CZ06' :'PF/SC-CZ06<KE',   # SC IDC
           'CZ07' :'PF/SC-CZ07<KE',   # SC IDC
           'CZ08' :'PF/SC-CZ08<KE',   # SC IDC
           'CZ09' :'PF/SC-CZ09<KE',   # SC IDC
           'CZ10' :'PF/SC-CZ10<KE',   # SC IDC
           'CZ11' :'PF/SC-CZ11<KE',   # SC IDC
           'CZ12' :'PF/SC-CZ12<KE',   # SC IDC
           'CZ13' :'PF/SC-CZ13<KE',   # SC IDC
           'CZ14' :'PF/SC-CZ14<KE',   # SC IDC
           'CZ15' :'PF/SC-CZ15<KE',   # SC IDC
           'CZ16' :'PF/SC-CZ16<KE',   # SC IDC
           'CZ17' :'PF/SC-CZ17<KE',   # SC IDC
           'CZ18' :'PF/SC-CZ18<KE',   # SC IDC
           }
    elif signalsTableName == 'signalsTable_TF_KC1D':
           sigTable = {
           # ODD and EVEN TF currents KC1D
           'TFODD' :'DA/C2-TFCI1B',   # TF current odd  # use for LOCA use after #67531
           'TFEVN' :'DA/C2-TFCI2B',   # TF current even # use for LOCA use after #67531
           #'TFEVN' :'DA/C2-ITFEVN',   # TF current odd # use before  #67531
           #'TFODD' :'DA/C2-ITFODD',   # TF current even # use before  #67531
           #'ITFB'  :'DA/C2-ITFB'  # TF current = [C2-ITFODD + C2-ITFEVN]/2
           }
    elif signalsTableName == 'signalsTable_TF_KC1Z':
           sigTable = {
           # ODD and EVEN TF currents KC1Z
           #'TFEVN' :'DA/C2Z-ITFEVN',   # TF current odd
           #'TFODD' :'DA/C2Z-ITFODD',   # TF current even
           'TFODD' :'DA/C2Z-TFCI1',   # TF current odd
           'TFEVN' :'DA/C2Z-TFCI2',   # TF current even
           }
    elif signalsTableName == 'signalsTable_TF_KC1E':
           sigTable = {
           # ODD and EVEN TF currents KC1E
           #'TFEVN' :'DA/CDE-ITFEV',   # TF current odd
           #'TFODD' :'DA/CDE-ITFOD',   # TF current even
           'TFODD' :'DA/C2E-TFCI1',   # TF current odd # use after #69199
           'TFEVN' :'DA/C2E-TFCI2',   # TF current even  # use after #69199
           }
    elif signalsTableName == 'signalsTable_TF_KC1E_CDE':
           sigTable = {
           # ODD and EVEN TF currents KC1E
           #'TFEVN' :'DA/CDE-ITFEV',   # TF current odd
           #'TFODD' :'DA/CDE-ITFOD',   # TF current even
           'TFODD' :'DA/CDE-TFCI1',   # TF current odd # use after #69199
           'TFEVN' :'DA/CDE-TFCI2',   # TF current even  # use after #69199
           }

    elif signalsTableName == 'signalsTable_EFIT':
           sigTable = {
                  # EFIT
                  'BPCA': 'PPF/EFIT/BPCA',  # simulated p-poloidal
                  'BPME': 'PPF/EFIT/BPME',  # MEASURED p-poloidal
                  'FLCA': 'PPF/EFIT/FLCA',  # simulated FLUX AND SADDLE
                  'FLME': 'PPF/EFIT/FLME',  # MEASURED FLUX AND SADDLE
                  'RBND': 'PPF/EFIT/RBND',  # r coordinate of boundary
                  'ZBND': 'PPF/EFIT/ZBND',  # z coordinate of boundary
                  'PSI':  'PPF/EFIT/PSI',  # 1089x989 psi gse solution  [33x33=1089]
                  'PSIR': 'PPF/EFIT/PSIR',  # 33 psi r grid
                  'PSIZ': 'PPF/EFIT/PSIZ',  # 33 psi r grid
                  'FBND': 'PPF/EFIT/FBND',  # psi at boundary
                  'RSIL': 'PPF/EFIT/RSIL',  # R inner lower strike (r of RSIGB)
                  'RSIU': 'PPF/EFIT/RSIU',  # R inner upper strike (r of ZSIGB)
                  'ZSIL': 'PPF/EFIT/ZSIL',  # Z inner lower strike (z of RSIGB)
                  'ZSIU': 'PPF/EFIT/ZSIU',  # Z inner upper strike (z of ZSIGB)
                  'RSOL': 'PPF/EFIT/RSOL',  # R outer lower strike (r of RSOGB)
                  'RSOU': 'PPF/EFIT/RSOU',  # R outer upper strike (r of ZSOGB)
                  'ZSOL': 'PPF/EFIT/ZSOL',  # Z outer lower strike (z of RSOGB)
                  'ZSOU': 'PPF/EFIT/ZSOU',  # Z outer upper strike (z of ZSOGB)
                  'NBND': 'PPF/EFIT/NBND',  # actual points of RBND,ZBND
                  'FAXS': 'PPF/EFIT/FAXS',  # psi at magnetic axis
                  'RMAG': 'PPF/EFIT/RMAG',  # r coordinate of magnetic axis
                  'ZMAG': 'PPF/EFIT/ZMAG',  # z coordinate of magnetic axis
           }


    elif signalsTableName == 'signalsTable_EHTR':
           sigTable = {
                  # EFIT
                  'BPCA': 'PPF/EHTR/BPCA',  # simulated p-poloidal
                  'BPME': 'PPF/EHTR/BPME',  # MEASURED p-poloidal
                  'FLCA': 'PPF/EHTR/FLCA',  # simulated FLUX AND SADDLE
                  'FLME': 'PPF/EHTR/FLME',  # MEASURED FLUX AND SADDLE
                  'RBND': 'PPF/EHTR/RBND',  # r coordinate of boundary
                  'ZBND': 'PPF/EHTR/ZBND',  # z coordinate of boundary
                  'PSI':  'PPF/EHTR/PSI',  # 1089x989 psi gse solution  [33x33=1089]
                  'PSIR': 'PPF/EHTR/PSIR',  # 33 psi r grid
                  'PSIZ': 'PPF/EHTR/PSIZ',  # 33 psi r grid
                  'FBND': 'PPF/EHTR/FBND',  # psi at boundary
                  'RSIL': 'PPF/EHTR/RSIL',  # R inner lower strike (r of RSIGB)
                  'RSIU': 'PPF/EHTR/RSIU',  # R inner upper strike (r of ZSIGB)
                  'ZSIL': 'PPF/EHTR/ZSIL',  # Z inner lower strike (z of RSIGB)
                  'ZSIU': 'PPF/EHTR/ZSIU',  # Z inner upper strike (z of ZSIGB)
                  'RSOL': 'PPF/EHTR/RSOL',  # R outer lower strike (r of RSOGB)
                  'RSOU': 'PPF/EHTR/RSOU',  # R outer upper strike (r of ZSOGB)
                  'ZSOL': 'PPF/EHTR/ZSOL',  # Z outer lower strike (z of RSOGB)
                  'ZSOU': 'PPF/EHTR/ZSOU',  # Z outer upper strike (z of ZSOGB)
                  'NBND': 'PPF/EHTR/NBND',  # actual points of RBND,ZBND
                  'FAXS': 'PPF/EHTR/FAXS',  # psi at magnetic axis
                  'RMAG': 'PPF/EHTR/RMAG',  # r coordinate of magnetic axis
                  'ZMAG': 'PPF/EHTR/ZMAG',  # z coordinate of magnetic axis
           }

    # ********************************************************
    # commissioning MAG JET restart 2018
    # ********************************************************
    elif signalsTableName ==   'signalsTable_KC1D_DCO_TP':
           sigTable = {
              # keys  Values
              # KC1D JPF Dicertor Coils (DC) OCT.3 divertorProbes1996
              'TP201' :'DA/C2-TP201',   # oct 3 tangential mod.7
              'TP202' :'DA/C2-TP202',   # oct 3 tangential mod.7
              'TP203' :'DA/C2-TP203',   # oct 3 tangential mod.7
              'TP204' :'DA/C2-TP204',   # oct 3 tangential mod.7
              'TP205' :'DA/C2-TP205',   # oct 3 tangential mod.7
              'TP206' :'DA/C2-TP206',   # oct 3 tangential mod.7
              'TP207' :'DA/C2-TP207',   # oct 3 tangential mod.7
              'TP208' :'DA/C2-TP208',   # oct 3 tangential mod.7
              'TP209' :'DA/C2-TP209',   # oct 3 tangential mod.7
              'TP210' :'DA/C2-TP210',   # oct 3 tangential mod.7
              'TP211' :'DA/C2-TP211',   # oct 3 tangential mod.7
              #
              #'TP201coeffTFE' :'DA/C2-TP201<TFE',   # TFE coefficients
              #'TP201coeffTFO' :'DA/C2-TP201<TFO',   # TFO coefficients
              # KC1D JPF Dicertor Coils (DC) OCT.7 divertorProbes1996
              'TP601' :'DA/C2-TP601',   # oct 7 tangential mod.19
              'TP602' :'DA/C2-TP602',   # oct 7 tangential mod.19
              'TP603' :'DA/C2-TP603',   # oct 7 tangential mod.19
              'TP604' :'DA/C2-TP604',   # oct 7 tangential mod.19
              'TP605' :'DA/C2-TP605',   # oct 7 tangential mod.19
              'TP606' :'DA/C2-TP606',   # oct 7 tangential mod.19
              'TP607' :'DA/C2-TP607',   # oct 7 tangential mod.19
              'TP608' :'DA/C2-TP608',   # oct 7 tangential mod.19
              'TP609' :'DA/C2-TP609',   # oct 7 tangential mod.19
              'TP610' :'DA/C2-TP610',   # oct 7 tangential mod.19
              'TP611' :'DA/C2-TP611',   # oct 7 tangential mod.19
           }
    elif signalsTableName ==   'signalsTable_KC1D_DCO_TN':
           sigTable = {
              # keys  Values
              # KC1D JPF Dicertor Coils (DC) OCT.3 divertorProbes1996
              'TN201' :'DA/C2-TN201',   # oct 3 normal mod.7
              'TN202' :'DA/C2-TN202',   # oct 3 normal mod.7
              'TN203' :'DA/C2-TN203',   # oct 3 normal mod.7
              'TN204' :'DA/C2-TN204',   # oct 3 normal mod.7
              'TN205' :'DA/C2-TN205',   # oct 3 normal mod.7
              'TN206' :'DA/C2-TN206',   # oct 3 normal mod.7
              'TN207' :'DA/C2-TN207',   # oct 3 normal mod.7
              'TN208' :'DA/C2-TN208',   # oct 3 normal mod.7
              'TN209' :'DA/C2-TN209',   # oct 3 normal mod.7
              'TN210' :'DA/C2-TN210',   # oct 3 normal mod.7
              'TN211' :'DA/C2-TN211',   # oct 3 normal mod.7
              # KC1D JPF Dicertor Coils (DC) OCT.7 divertorProbes1996
              'TN601' :'DA/C2-TN601',   # oct 7 normal mod.19
              'TN602' :'DA/C2-TN602',   # oct 7 normal mod.19
              'TN603' :'DA/C2-TN603',   # oct 7 normal mod.19
              'TN604' :'DA/C2-TN604',   # oct 7 normal mod.19
              'TN605' :'DA/C2-TN605',   # oct 7 normal mod.19
              'TN606' :'DA/C2-TN606',   # oct 7 normal mod.19
              'TN607' :'DA/C2-TN607',   # oct 7 normal mod.19
              'TN608' :'DA/C2-TN608',   # oct 7 normal mod.19
              'TN609' :'DA/C2-TN609',   # oct 7 normal mod.19
              'TN610' :'DA/C2-TN610',   # oct 7 normal mod.19
              'TN611' :'DA/C2-TN611',   # oct 7 normal mod.19
           }
    elif signalsTableName ==   'signalsTable_KC1Z_DCO_TP':
           sigTable = {
              # keys  Values
              # KC1Z JPF Dicertor Coils (DC) OCT.3 divertorProbes1996
              'TP201' :'DA/C2Z-TP201',   # oct 3 tangential mod.7
              #'TP202' :'DA/C2Z-TP202',   # oct 3 tangential mod.7
              #'TP203' :'DA/C2Z-TP203',   # oct 3 tangential mod.7
              #'TP204' :'DA/C2Z-TP204',   # oct 3 tangential mod.7
              #'TP205' :'DA/C2Z-TP205',   # oct 3 tangential mod.7
              'TP206' :'DA/C2Z-TP206',   # oct 3 tangential mod.7
              #'TP207' :'DA/C2Z-TP207',   # oct 3 tangential mod.7
              'TP208' :'DA/C2Z-TP208',   # oct 3 tangential mod.7
              #'TP209' :'DA/C2Z-TP209',   # oct 3 tangential mod.7
              'TP210' :'DA/C2Z-TP210',   # oct 3 tangential mod.7
              #'TP211' :'DA/C2Z-TP211',   # oct 3 tangential mod.7
              #
              #'TP201coeffTFE' :'DA/C2Z-TP201<TFE',   # TFE coefficients
              #'TP201coeffTFO' :'DA/C2Z-TP201<TFO',   # TFO coefficients
              # KC1Z JPF Dicertor Coils (DC) OCT.7 divertorProbes1996
              'TP601' :'DA/C2Z-TP601',   # oct 7 tangential mod.19
              'TP602' :'DA/C2Z-TP602',   # oct 7 tangential mod.19
              'TP603' :'DA/C2Z-TP603',   # oct 7 tangential mod.19
              'TP604' :'DA/C2Z-TP604',   # oct 7 tangential mod.19
              #'TP605' :'DA/C2Z-TP605',   # oct 7 tangential mod.19
              'TP606' :'DA/C2Z-TP606',   # oct 7 tangential mod.19
              #'TP607' :'DA/C2Z-TP607',   # oct 7 tangential mod.19
              'TP608' :'DA/C2Z-TP608',   # oct 7 tangential mod.19
              'TP609' :'DA/C2Z-TP609',   # oct 7 tangential mod.19
              'TP610' :'DA/C2Z-TP610',   # oct 7 tangential mod.19
              'TP611' :'DA/C2Z-TP611',   # oct 7 tangential mod.19
           }
    elif signalsTableName ==   'signalsTable_KC1Z_DCO_TN':
           sigTable = {
              # keys  Values
              # KC1Z JPF Dicertor Coils (DC) OCT.3 divertorProbes1996
              #'TN201' :'DA/C2Z-TN201',   # oct 3 normal mod.7
              #'TN202' :'DA/C2Z-TN202',   # oct 3 normal mod.7
              'TN203' :'DA/C2Z-TN203',   # oct 3 normal mod.7
              #'TN204' :'DA/C2Z-TN204',   # oct 3 normal mod.7
              #'TN205' :'DA/C2Z-TN205',   # oct 3 normal mod.7
              'TN206' :'DA/C2Z-TN206',   # oct 3 normal mod.7
              'TN207' :'DA/C2Z-TN207',   # oct 3 normal mod.7
              #'TN208' :'DA/C2Z-TN208',   # oct 3 normal mod.7
              'TN209' :'DA/C2Z-TN209',   # oct 3 normal mod.7
              #'TN210' :'DA/C2Z-TN210',   # oct 3 normal mod.7
              'TN211' :'DA/C2Z-TN211',   # oct 3 normal mod.7
              #
              # KC1Z JPF Dicertor Coils (DC) OCT.7 divertorProbes1996
              'TN601' :'DA/C2Z-TN601',   # oct 7 normal mod.19
              'TN602' :'DA/C2Z-TN602',   # oct 7 normal mod.19
              'TN603' :'DA/C2Z-TN603',   # oct 7 normal mod.19
              'TN604' :'DA/C2Z-TN604',   # oct 7 normal mod.19
              #'TN605' :'DA/C2Z-TN605',   # oct 7 normal mod.19
              #'TN606' :'DA/C2Z-TN606',   # oct 7 normal mod.19
              #'TN607' :'DA/C2Z-TN607',   # oct 7 normal mod.19
              'TN608' :'DA/C2Z-TN608',   # oct 7 normal mod.19
              'TN609' :'DA/C2Z-TN609',   # oct 7 normal mod.19
              'TN610' :'DA/C2Z-TN610',   # oct 7 normal mod.19
              'TN611' :'DA/C2Z-TN611',   # oct 7 normal mod.19
           }
    elif signalsTableName ==   'signalsTable_KC1D_P8AB':
           sigTable = {
              # keys  Values
              # KC1D JPF Poloidal Limiter Coils OCT.8
              'P801A' :'DA/C2-P801A',   # oct 8
              'P802A' :'DA/C2-P802A',   # oct 8
              'P803A' :'DA/C2-P803A',   # oct 8
              'P804A' :'DA/C2-P804A',   # oct 8
              'P805A' :'DA/C2-P805A',   # oct 8
              'P806A' :'DA/C2-P806A',   # oct 8
              'P807A' :'DA/C2-P807A',   # oct 8
                  #
              'P801B' :'DA/C2-P801B',   # oct 8
              'P802B' :'DA/C2-P802B',   # oct 8
              'P803B' :'DA/C2-P803B',   # oct 8
              'P804B' :'DA/C2-P804B',   # oct 8
              'P805B' :'DA/C2-P805B',   # oct 8
              'P806B' :'DA/C2-P806B',   # oct 8
              'P807B' :'DA/C2-P807B',   # oct 8
           }
    elif signalsTableName ==   'signalsTable_KC1Z_P8AB':
           sigTable = {
              # keys  Values
              # KC1D JPF Poloidal Limiter Coils OCT.8
              'P801A' :'DA/C2Z-P801A',   # oct 8
              'P802A' :'DA/C2Z-P802A',   # oct 8
              'P803A' :'DA/C2Z-P803A',   # oct 8
              'P804A' :'DA/C2Z-P804A',   # oct 8
              'P805A' :'DA/C2Z-P805A',   # oct 8
              'P806A' :'DA/C2Z-P806A',   # oct 8
              'P807A' :'DA/C2Z-P807A',   # oct 8
                  #
              'P801B' :'DA/C2Z-P801B',   # oct 8
              'P802B' :'DA/C2Z-P802B',   # oct 8
              'P803B' :'DA/C2Z-P803B',   # oct 8
              'P804B' :'DA/C2Z-P804B',   # oct 8
              'P805B' :'DA/C2Z-P805B',   # oct 8
              'P806B' :'DA/C2Z-P806B',   # oct 8
              'P807B' :'DA/C2Z-P807B',   # oct 8
           }
    elif signalsTableName ==   'signalsTable_KC1D_FL':
           sigTable = {
              # keys  Values
              # KC1D JPF FULL FLUX LOOPS
              'FL05'  :'DA/C2-FL05',
              'FL11'  :'DA/C2-FL11',
              'FLRRU' :'DA/C2-FLRRU',
              'FLRRL' :'DA/C2-FLRRL',
              'FLD2'  :'DA/C2-FLD2',
              'FLD3'  :'DA/C2-FLD3',
              #'FLD4A'  :'DA/C2-FLD4A',
              #'FLD4B'  :'DA/C2-FLD4B',
           }
    elif signalsTableName ==   'signalsTable_KC1Z_FL':
           sigTable = {
              # keys  Values
              # KC1Z JPF FULL FLUX LOOPS
              'FL05'  :'DA/C2Z-FL05',
              'FL11'  :'DA/C2Z-FL11',
              'FLRRU' :'DA/C2Z-FLRRU',
              'FLRRL' :'DA/C2Z-FLRRL',
              'FLD2'  :'DA/C2Z-FLD2',
              'FLD3'  :'DA/C2Z-FLD3',
           }
    elif signalsTableName ==   'signalsTable_KC1E_FL':
           sigTable = {
              # keys  Values
              # KC1E JPF FULL FLUX LOOPS
              'FL05'  :'DA/C2E-FL05',
              'FL11'  :'DA/C2E-FL11',
              'FLRRU' :'DA/C2E-FLRRU',
              'FLRRL' :'DA/C2E-FLRRL',
              'FLD2'  :'DA/C2E-FLD2',
              'FLD3'  :'DA/C2E-FLD3',
           }
    elif signalsTableName ==   'signalsTable_KC1D_UP':
           sigTable = {
              # keys  Values
              # KC1D JPF upper coils tangential
              'UP801'  :'DA/C2-UP801',
              'UP802'  :'DA/C2-UP802',
              'UP803'  :'DA/C2-UP803',
           }
    elif signalsTableName ==   'signalsTable_KC1D_UN':
           sigTable = {
              # keys  Values
              # KC1D JPF upper coils normal
              'UN801'  :'DA/C2-UN801',
              'UN802'  :'DA/C2-UN802',
              'UN803'  :'DA/C2-UN803',
              'UN804'  :'DA/C2-UN804',
             # 'UN805'  :'DA/C2-UN805',
           }
    elif signalsTableName ==   'signalsTable_KC1Z_UP':
           sigTable = {
              # keys  Values
              # KC1Z JPF upper coils tangential
              'UP801'  :'DA/C2Z-UP801',
              'UP802'  :'DA/C2Z-UP802',
              'UP803'  :'DA/C2Z-UP803',
           }
    elif signalsTableName ==   'signalsTable_KC1Z_UN':
           sigTable = {
              # keys  Values
              # KC1D JPF upper coils normal
              'UN801'  :'DA/C2Z-UN801',
              'UN802'  :'DA/C2Z-UN802',
              'UN803'  :'DA/C2Z-UN803',
              'UN804'  :'DA/C2Z-UN804',
              #'UN805'  :'DA/C2Z-UN805',
           }
    elif signalsTableName ==   'signalsTable_KC1E_PP':
           sigTable = {
              # keys  Values
              # KC1E JPF poloidal limiter coils oct.4
              'PP400'  :'DA/C2E-PP400',
              'PP401'  :'DA/C2E-PP401',
              'PP402'  :'DA/C2E-PP402',
              'PP403'  :'DA/C2E-PP403',
              'PP404'  :'DA/C2E-PP404',
              'PP405'  :'DA/C2E-PP405',
              'PP406'  :'DA/C2E-PP406',
              'PP407'  :'DA/C2E-PP407',
               # KC1E JPF poloidal limiter coils oct.8
              'PP800'  :'DA/C2E-PP800',
              'PP801'  :'DA/C2E-PP801',
              'PP802'  :'DA/C2E-PP802',
              'PP803'  :'DA/C2E-PP803',
              'PP804'  :'DA/C2E-PP804',
              'PP805'  :'DA/C2E-PP805',
              'PP806'  :'DA/C2E-PP806',
              'PP807'  :'DA/C2E-PP807',
           }
    elif signalsTableName ==   'signalsTable_KC1E_PN':
           sigTable = {
              # keys  Values
              # KC1E JPF poloidal limiter coils oct.4
              'PN400'  :'DA/C2E-PN400',
              'PN401'  :'DA/C2E-PN401',
              'PN402'  :'DA/C2E-PN402',
              'PN403'  :'DA/C2E-PN403',
              'PN404'  :'DA/C2E-PN404',
              'PN405'  :'DA/C2E-PN405',
              'PN406'  :'DA/C2E-PN406',
              'PN407'  :'DA/C2E-PN407',
               # KC1E JPF poloidal limiter coils oct.8
              'PN800'  :'DA/C2E-PN800',
              'PN801'  :'DA/C2E-PN801',
              'PN802'  :'DA/C2E-PN802',
              'PN803'  :'DA/C2E-PN803',
              'PN804'  :'DA/C2E-PN804',
              'PN805'  :'DA/C2E-PN805',
              'PN806'  :'DA/C2E-PN806',
              'PN807'  :'DA/C2E-PN807',
           }
    elif signalsTableName ==   'signalsTable_KC1D_IC':
           sigTable = {
              # keys  Values
              # KC1D JPF Inner coils oct.8
              'I801'  :'DA/C2-I801',
              'I802'  :'DA/C2-I802',
              'I803'  :'DA/C2-I803'
           }
    elif signalsTableName ==   'signalsTable_KC1Z_IC':
           sigTable = {
              # keys  Values
              # KC1Z JPF Inner coils oct.8
              'I801'  :'DA/C2Z-I801',
              'I802'  :'DA/C2Z-I802',
              'I803'  :'DA/C2Z-I803'
           }
    elif signalsTableName ==   'signalsTable_KC1E_UPE':
           sigTable = {
              # keys  Values
              # KC1E JPF Inner coils oct.4
              'UP4E1'  :'DA/C2E-UP4E1',
              'UP4E2'  :'DA/C2E-UP4E2',
              'UP4E3'  :'DA/C2E-UP4E3',
              'UP4E4'  :'DA/C2E-UP4E4',
              # KC1E JPF Inner coils oct.8
              'UP8E1'  :'DA/C2E-UP8E1',
              'UP8E2'  :'DA/C2E-UP8E2',
              'UP8E3'  :'DA/C2E-UP8E3',
              'UP8E4'  :'DA/C2E-UP8E4',
           }
    elif signalsTableName ==   'signalsTable_KC1E_UNE':
           sigTable = {
              # keys  Values
              # KC1E JPF Inner coils oct.4
              'UN4E1'  :'DA/C2E-UN4E1',
              'UN4E2'  :'DA/C2E-UN4E2',
              'UN4E3'  :'DA/C2E-UN4E3',
              'UN4E4'  :'DA/C2E-UN4E4',
              # KC1E JPF Inner coils oct.8
              'UN8E1'  :'DA/C2E-UN8E1',
              'UN8E2'  :'DA/C2E-UN8E2',
              'UN8E3'  :'DA/C2E-UN8E3',
              'UN8E4'  :'DA/C2E-UN8E4',
           }
    elif signalsTableName ==   'signalsTable_KC1D_BSL':
           sigTable = {
              # keys  Values
              # KC1D JPF Big Saddle Loops oct.3
              'S3U'  :'DA/C2-S3U',
              'S3L'  :'DA/C2-S3L',
              'S3O'  :'DA/C2-S3O',
              'S3I'  :'DA/C2-S3I',
              # KC1D JPF Big Saddle Loops oct.7
              'S7U'  :'DA/C2-S7U',
              'S7L'  :'DA/C2-S7L',
              'S7O'  :'DA/C2-S7O',
              'S7I'  :'DA/C2-S7I',
           }
    elif signalsTableName ==   'signalsTable_KC1Z_BSL':
           sigTable = {
              # keys  Values
              # KC1Z JPF Big Saddle Loops oct.3
              'S3U'  :'DA/C2Z-S3U',
              'S3L'  :'DA/C2Z-S3L',
              'S3O'  :'DA/C2Z-S3O',
              'S3I'  :'DA/C2Z-S3I',
              # KC1Z JPF Big Saddle Loops oct.7
              'S7U'  :'DA/C2Z-S7U',
              'S7L'  :'DA/C2Z-S7L',
              'S7O'  :'DA/C2Z-S7O',
              'S7I'  :'DA/C2Z-S7I',
           }
    elif signalsTableName ==   'signalsTable_KC1E_BSL':
           sigTable = {
              # keys  Values
              # KC1Z JPF Big Saddle Loops oct.7
              'S7U'  :'DA/C2E-S7U',
              'S7L'  :'DA/C2E-S7L',
              #'S7O'  :'DA/C2E-S7O',
              #'S7I'  :'DA/C2E-S7I',
           }
    elif signalsTableName ==   'signalsTable_KC1D_OPL':
           sigTable = {
              # keys  Values
              # KC1D JPF Outer poloidal limiter coils oct.4
              'P401'  :'DA/C2-P401',
              'P402'  :'DA/C2-P402',
              'P403'  :'DA/C2-P403',
              'P404'  :'DA/C2-P404',
              'P405'  :'DA/C2-P405',
              'P406'  :'DA/C2-P406',
              'P407'  :'DA/C2-P407',
               # KC1D JPF Outer poloidal limiter coils oct.8
              'P801'  :'DA/C2-P801',
              'P802'  :'DA/C2-P802',
              'P803'  :'DA/C2-P803',
              'P804'  :'DA/C2-P804',
              'P805'  :'DA/C2-P805',
              'P806'  :'DA/C2-P806',
              'P807'  :'DA/C2-P807',
           }
    elif signalsTableName ==   'signalsTable_KC1Z_OPL':
           sigTable = {
              # keys  Values
              # KC1Z JPF Outer poloidal limiter coils oct.4
              'P402'  :'DA/C2Z-P402',
               # KC1Z JPF Outer poloidal limiter coils oct.8
              'P802'  :'DA/C2Z-P802',
           }
    elif signalsTableName ==   'signalsTable_KC1E_DTCP':
           sigTable = {
              # keys  Values
              # KC1E JPF Divertor Target Coils Tangential mod.3
              'TP131'  :'DA/C2E-TP131',
              'TP132'  :'DA/C2E-TP132',
              'TP133'  :'DA/C2E-TP133',
              'TP134'  :'DA/C2E-TP134',
              'TP135'  :'DA/C2E-TP135',
              'TP136'  :'DA/C2E-TP136',
              'TP137'  :'DA/C2E-TP137',
              # KC1E JPF Divertor Target Coils Tangential mod.5
              'TP251'  :'DA/C2E-TP251',
              'TP252'  :'DA/C2E-TP252',
              'TP253'  :'DA/C2E-TP253',
              'TP254'  :'DA/C2E-TP254',
              'TP255'  :'DA/C2E-TP255',
              'TP256'  :'DA/C2E-TP256',
              'TP257'  :'DA/C2E-TP257',
           }
    elif signalsTableName ==   'signalsTable_KC1E_DTCN':
           sigTable = {
              # keys  Values
              # KC1E JPF Divertor Target Coils Normal mod.3
              'TN131'  :'DA/C2E-TN131',
              'TN132'  :'DA/C2E-TN132',
              'TN133'  :'DA/C2E-TN133',
              'TN134'  :'DA/C2E-TN134',
              'TN135'  :'DA/C2E-TN135',
              'TN136'  :'DA/C2E-TN136',
              'TN137'  :'DA/C2E-TN137',
              # KC1E JPF Divertor Target Coils Normal mod.5
              'TN251'  :'DA/C2E-TN251',
              'TN252'  :'DA/C2E-TN252',
              'TN253'  :'DA/C2E-TN253',
              'TN254'  :'DA/C2E-TN254',
              'TN255'  :'DA/C2E-TN255',
              'TN256'  :'DA/C2E-TN256',
              'TN257'  :'DA/C2E-TN257',
           }
    elif signalsTableName ==   'signalsTable_KC1D_DSL':
           sigTable = {
              # keys  Values
              # KC1D JPF Divertor Saddle loops oct.1-2-3
              'TS201'  :'DA/C2-TS201',
              'TS202'  :'DA/C2-TS202',
              'TS203'  :'DA/C2-TS203',
              'TS204'  :'DA/C2-TS204',
              'TS205'  :'DA/C2-TS205',
              'TS206'  :'DA/C2-TS206',
              'TS207'  :'DA/C2-TS207',
              'TS208'  :'DA/C2-TS208',
              'TS209'  :'DA/C2-TS209',
              'TS210'  :'DA/C2-TS210',
              'TS211'  :'DA/C2-TS211',
              'TS212'  :'DA/C2-TS212',
              # KC1D JPF Divertor Saddle loops oct.4-5-6
              'TS601'  :'DA/C2-TS601',
              'TS602'  :'DA/C2-TS602',
              'TS603'  :'DA/C2-TS603',
              'TS604'  :'DA/C2-TS604',
              'TS605'  :'DA/C2-TS605',
              'TS606'  :'DA/C2-TS606',
              'TS607'  :'DA/C2-TS607',
              'TS608'  :'DA/C2-TS608',
              'TS609'  :'DA/C2-TS609',
              'TS610'  :'DA/C2-TS610',
              'TS611'  :'DA/C2-TS611',
              'TS612'  :'DA/C2-TS612',
           }
    elif signalsTableName ==   'signalsTable_KC1Z_DSL':
           sigTable = {
              # keys  Values
              # KC1Z JPF Divertor Saddle loops oct.1-2-3
              #'TS201'  :'DA/C2Z-TS201',
              'TS202'  :'DA/C2Z-TS202',
              'TS203'  :'DA/C2Z-TS203',
              'TS204'  :'DA/C2Z-TS204',
              #'TS205'  :'DA/C2Z-TS205',
              'TS206'  :'DA/C2Z-TS206',
              'TS207'  :'DA/C2Z-TS207',
              'TS208'  :'DA/C2Z-TS208',
              'TS209'  :'DA/C2Z-TS209',
              'TS210'  :'DA/C2Z-TS210',
              'TS211'  :'DA/C2Z-TS211',
              'TS212'  :'DA/C2Z-TS212',
              # KC1Z JPF Divertor Saddle loops oct.4-5-6
              'TS601'  :'DA/C2Z-TS601',
              'TS602'  :'DA/C2Z-TS602',
              'TS603'  :'DA/C2Z-TS603',
              'TS604'  :'DA/C2Z-TS604',
              'TS605'  :'DA/C2Z-TS605',
              'TS606'  :'DA/C2Z-TS606',
              'TS607'  :'DA/C2Z-TS607',
              'TS608'  :'DA/C2Z-TS608',
              #'TS609'  :'DA/C2Z-TS609',
              #'TS610'  :'DA/C2Z-TS610',
              'TS611'  :'DA/C2Z-TS611',
              'TS612'  :'DA/C2Z-TS612',
           }
    elif signalsTableName ==   'signalsTable_KC1E_EVPU':
           sigTable = {
              # keys  Values
              # KC1E JPF Ex-Vessel Pick-Up coils oct.3
              'EC31V'  :'DA/C2E-EC31V',
              'EC32V'  :'DA/C2E-EC32V',
              'EC33V'  :'DA/C2E-EC33V',
               # KC1E JPF Ex-Vessel Pick-Up coils oct.8
              'EC81V'  :'DA/C2E-EC81V',
              'EC82V'  :'DA/C2E-EC82V',
              'EC83V'  :'DA/C2E-EC83V',
           }
    elif signalsTableName ==   'signalsTable_KC1E_EVH':
           sigTable = {
              # keys  Values
              # KC1E JPF Ex-Vessel Hall coils oct.3
              'EH31V'  :'DA/C2E-EH31V',
              'EH32V'  :'DA/C2E-EH32V',
              'EH33V'  :'DA/C2E-EH33V',
               # KC1E JPF Ex-Vessel Hall coils oct.8
              'EH81V'  :'DA/C2E-EH81V',
              'EH82V'  :'DA/C2E-EH82V',
              'EH83V'  :'DA/C2E-EH83V',
           }
    elif signalsTableName ==   'signalsTable_KC1E_EVFL':
           sigTable = {
              # keys  Values
              # KC1E JPF Ex-Vessel Flux loop oct.3
              'EFL31'  :'DA/C2E-EFL31',
              'EFL32'  :'DA/C2E-EFL32',
              'EFL33'  :'DA/C2E-EFL33',
               # KC1E JPF Ex-Vessel Flux Loop oct.8
              'EFL81'  :'DA/C2E-EFL81',
              'EFL82'  :'DA/C2E-EFL82',
              'EFL83'  :'DA/C2E-EFL83',
           }
    elif signalsTableName ==   'signalsTable_KC1E_EVCH':
           sigTable = {
              # keys  Values
              # KC1E JPF Ex-Vessel Collar Hall probe
              'EHC4V'  :'DA/C2E-EHC4V',
              'EHC4R'  :'DA/C2E-EHC4R',
               # KC1E JPF Ex-Vessel Collar Pick Up coils
              'ECC4V'  :'DA/C2E-ECC4V',
              'ECC4R'  :'DA/C2E-ECC4R',
           }
    elif signalsTableName == 'signalsTable_KC1D_LOCA':
           sigTable = {
           # keys  Values
           # KC1D JPF saddle loops
           'SX01' :'DA/C2-SX01',
           'SX14' :'DA/C2-SX14',
           'SY01' :'DA/C2-SY01',
           'SY14' :'DA/C2-SY14',
           'S101' :'DA/C2-S101',
           'S114' :'DA/C2-S114',
           'S501' :'DA/C2-S501',
           'S514' :'DA/C2-S514',
           }
    elif signalsTableName == 'signalsTable_KC1Z_LOCA':
           sigTable = {
           # keys  Values
           # KC1Z JPF saddle loops
           'SX01' :'DA/C2Z-SX01',
           'SX14' :'DA/C2Z-SX14',
           'SY01' :'DA/C2Z-SY01',
           'SY14' :'DA/C2Z-SY14',
           'S101' :'DA/C2Z-S101',
           'S114' :'DA/C2Z-S114',
           'S501' :'DA/C2Z-S501',
           'S514' :'DA/C2Z-S514',
           }
    elif signalsTableName == 'signalsTable_KC1E_LOCA':
           sigTable = {
           # keys  Values
           # KC1E JPF saddle loops
           'S101' :'DA/C2E-S101',
           'S301' :'DA/C2E-S301',
           'S501' :'DA/C2E-S501',
           'S701' :'DA/C2E-S701',
           'S114' :'DA/C2E-S114',
           'S314' :'DA/C2E-S314',
           'S514' :'DA/C2E-S514',
           'S714' :'DA/C2E-S714',
           }
    elif signalsTableName == 'signalsTable_KC1E_CDE_LOCA':
           sigTable = {
           # keys  Values
           # KC1E JPF saddle loops 5kHz
           'S101' :'DA/CDE-S101',
           'S301' :'DA/CDE-S301',
           'S501' :'DA/CDE-S501',
           'S701' :'DA/CDE-S701',
           'S114' :'DA/CDE-S114',
           'S314' :'DA/CDE-S314',
           'S514' :'DA/CDE-S514',
           'S714' :'DA/CDE-S714',
           }
    elif signalsTableName == 'signalsTable_extra':
           sigTable = {
           # keys  Values
           # KC1Z
           'LOCA_KC1Z' :'DA/C2Z-LOCA>',
           'MHDF_KC1Z' :'DA/C2Z-MHDF>',
           'MHDG_KC1Z' :'DA/C2Z-MHDG>',
            # keys  Values
           # KC1E 500Hz
           'LOCA_KC1E' :'DA/C2E-LOCA:001',
           #'LOCA_KC1E' :'DA/C2E-LOCA:002',
           # KC1E 5lHz
           'LOCA_KC1E_CDE' :'DA/CDE-LOCA:001',
           #'LOCA_KC1E_CDE' :'DA/CDE-LOCA:002',
           # keys  Values
           # KC1D
           'LOCA_KC1D' :'DA/C2-LOCA',
           #'LOCA_KC1D' :'DA/C2-LOCA:001',
           #'LOCA_KC1D' :'DA/C2-LOCA:002',
           'MHD_KC1D'  :'DA/C2-MHD',
           'MHDF_KC1D' :'DA/C2-MHDF',
           'MHDG_KC1D' :'DA/C2-MHDG'
           }
    elif signalsTableName == 'signalsTable_PlasmaCurrentMoments':
           sigTable = {
           # keys  Values
           'IPLA_KC1D' :'DA/C2-IPLA',
           'PVPM_KC1D' :'DA/C2-PVPM',
           'PRPM_KC1D' :'DA/C2-PRPM',
            # keys  Values
           'IPLA_KC1Z' :'DA/C2Z-IPLA>',
           'PVPM_KC1Z' :'DA/C2Z-PVPM>',
           'PRPM_KC1Z' :'DA/C2Z-PRPM>',
            # keys  Values
           'IPLA_KC1E' :'DA/C2E-IPLA:001',
           'PVPM_KC1E' :'DA/CDE-PVPM',
           'PRPM_KC1E' :'DA/CDE-PRPM',
           }
    # SHAPE VISUALIZER
    elif signalsTableName == 'signalsTable_XLOC':
           sigTable = {
              # keys  Values
              # XLOC GAPS,STRIKE POINTS and LIMITER/DIVERTED INDICATOR (CTYPE)
           # GAPS
           'LOG'    :  'PF/SC-LOG<XS'       ,     # 'Lower out gap XLOC'
           'ROG'    :  'PF/SC-ROG<XS'       ,     # 'Outer gap XLOC'
           'ZUP'    :  'PF/SC-ZUP<XS'       ,     # 'Top inner gap XLOC'
           'RIG'    :  'PF/SC-RIG<XS'       ,     # 'Inner gap XLOC'
           #'SOG'    :  'PF/SC-SOG<XS'       ,     #    'TBD'
           #'SIG'    :  'PF/SC-SIG<XS'       ,     #    'TBD'
           'TOG1'   :  'PF/SC-TOG<XS:001'   ,     # 'Top gap XLOC comp 1'
           'TOG2'   :  'PF/SC-TOG<XS:002'   ,     # 'Top gap XLOC comp 2'
           'TOG3'   :  'PF/SC-TOG<XS:003'   ,     # 'Top gap XLOC comp 3'
           'TOG4'   :  'PF/SC-TOG<XS:004'   ,     # 'Top gap XLOC comp 4'
           'TOG5'   :  'PF/SC-TOG<XS:005'   ,     # 'Top gap XLOC comp 5'
           'GAP2'   :  'PF/SC-GAP<XS:002'   ,     # 'XLOC gap 2'
           'GAP3'   :  'PF/SC-GAP<XS:003'   ,     # 'XLOC gap 3'
           'GAP4'   :  'PF/SC-GAP<XS:004'   ,     # 'XLOC gap 4'
           'GAP6'   :  'PF/SC-GAP<XS:006'   ,     # 'XLOC gap 6'
           'GAP7'   :  'PF/SC-GAP<XS:007'   ,     # 'XLOC gap 7'
           'GAP17'  :  'PF/SC-GAP<XS:017'   ,     # 'XSC gap 1'
           'GAP18'  :  'PF/SC-GAP<XS:018'   ,     # 'XSC gap 2'
           'GAP19'  :  'PF/SC-GAP<XS:019'   ,     # 'XSC gap 3'
           'GAP20'  :  'PF/SC-GAP<XS:020'   ,     # 'XSC gap 4'
           'GAP21'  :  'PF/SC-GAP<XS:021'   ,     # 'XSC gap 5'
           'GAP22'  :  'PF/SC-GAP<XS:022'   ,     # 'XSC gap 6'
           'GAP23'  :  'PF/SC-GAP<XS:023'   ,     # 'XSC gap 7'
           'GAP24'  :  'PF/SC-GAP<XS:024'   ,     # 'XSC gap 8'
           'GAP25'  :  'PF/SC-GAP<XS:025'   ,     # 'XSC gap 9'
           'GAP26'  :  'PF/SC-GAP<XS:026'   ,     # 'XSC gap 10'
           'GAP27'  :  'PF/SC-GAP<XS:027'   ,     # 'XSC gap 11'
           'GAP28'  :  'PF/SC-GAP<XS:028'   ,     # 'XSC gap 12'
           'GAP29'  :  'PF/SC-GAP<XS:029'   ,     # 'XSC gap 13'
           'GAP30'  :  'PF/SC-GAP<XS:030'   ,     # 'XSC gap 14'
           'GAP31'  :  'PF/SC-GAP<XS:031'   ,     # 'XSC gap 15'
           'GAP32'  :  'PF/SC-GAP<XS:032'   ,     # 'XSC gap 16'
           #  STRIKE POINTS
           'RSOGB'  :    'PF/SC-RSOGB<XS'  , #   'In-tgt st-pt XLOC'
           'RSIGB'  :    'PF/SC-RSIGB<XS'  , #   'In-side st-pt XLOC'
           'ZSOGB'  :    'PF/SC-ZSOGB<XS'  , #   'Out-tgt st-pt XLOC'
           'ZSIGB'  :    'PF/SC-ZSIGB<XS'  , #   'Out-side st-pt XLOC'
           'WLBSRP'  :   'PF/SC-WLBSRP<XS'  , #   '
       # X POINT
           'RX'     :    'PF/SC-RXGB<XS'   , #   'X-point radius XLOC'
           'ZX'     :    'PF/SC-ZXGB<XS'   , #   'X-point height XLOC'
           'CTYPE'  :    'PF/SC-CTYPE<XS'  , #   'XLOC config type: -1 is xpoint'
    }
    elif signalsTableName == 'signalsTable_SC':
           sigTable = {
              # keys  Values
           'IPRIM'  :  'PF/SC-IP1<MS'  ,  #       'P1 Ext curr'
           'IP4T'   :  'PF/SC-IP4<MS'  ,  #       'P4 total curr'
           'IP4IM'  :  'PF/SC-IIM<MS'  ,  #       'P4 Imb curr'
           'IPFX'   :  'PF/SC-IFX<MS'  ,  #       'PFX curr'
           'ISHP'   :  'PF/SC-ISH<MS'  ,  #       'Shaping curr'
           'ID1'    :  'PF/SC-ID1<MS'  ,  #       'D1 curr'
           'ID2'    :  'PF/SC-ID2<MS'  ,  #       'D2 curr'
           'ID3'    :  'PF/SC-ID3<MS'  ,  #       'D3 curr'
           'ID4'    :  'PF/SC-ID4<MS'  ,  #       'D4 curr'
           'IP'     :  'PF/SC-IP<MS'   ,  #       'Plasma curr'
           }

    elif signalsTableName == 'signalsTable_WALLS':
           sigTable = {
              # keys  Values
              'IWLGR01':'PF/WA-IWLGR<XS:001',
              'IWLGR02':'PF/WA-IWLGR<XS:002',
              'IWLGR03':'PF/WA-IWLGR<XS:003',
              'IWLGR04':'PF/WA-IWLGR<XS:004',
              'IWLGR05':'PF/WA-IWLGR<XS:005',
              'IWLGR06':'PF/WA-IWLGR<XS:006',
              'IWLGR07':'PF/WA-IWLGR<XS:007',
              'IWLGR08':'PF/WA-IWLGR<XS:008',
              'IWLGR09':'PF/WA-IWLGR<XS:009',
              'IWLGR10':'PF/WA-IWLGR<XS:010',
              'IWLGR11':'PF/WA-IWLGR<XS:011',
              'IWLGR12':'PF/WA-IWLGR<XS:012',
              'IWLGR13':'PF/WA-IWLGR<XS:013',
              'IWLGR14':'PF/WA-IWLGR<XS:014',
              'IWLGR15':'PF/WA-IWLGR<XS:015',
              'IWLGR16':'PF/WA-IWLGR<XS:016',
              'IWLGR17':'PF/WA-IWLGR<XS:017',
              'IWLGR18':'PF/WA-IWLGR<XS:018',
              'IWLGR19':'PF/WA-IWLGR<XS:019',
              #
              'IWLGZ01':'PF/WA-IWLGZ<XS:001',
              'IWLGZ02':'PF/WA-IWLGZ<XS:002',
              'IWLGZ03':'PF/WA-IWLGZ<XS:003',
              'IWLGZ04':'PF/WA-IWLGZ<XS:004',
              'IWLGZ05':'PF/WA-IWLGZ<XS:005',
              'IWLGZ06':'PF/WA-IWLGZ<XS:006',
              'IWLGZ07':'PF/WA-IWLGZ<XS:007',
              'IWLGZ08':'PF/WA-IWLGZ<XS:008',
              'IWLGZ09':'PF/WA-IWLGZ<XS:009',
              'IWLGZ10':'PF/WA-IWLGZ<XS:010',
              'IWLGZ11':'PF/WA-IWLGZ<XS:011',
              'IWLGZ12':'PF/WA-IWLGZ<XS:012',
              'IWLGZ13':'PF/WA-IWLGZ<XS:013',
              'IWLGZ14':'PF/WA-IWLGZ<XS:014',
              'IWLGZ15':'PF/WA-IWLGZ<XS:015',
              'IWLGZ16':'PF/WA-IWLGZ<XS:016',
              'IWLGZ17':'PF/WA-IWLGZ<XS:017',
              'IWLGZ18':'PF/WA-IWLGZ<XS:018',
              'IWLGZ19':'PF/WA-IWLGZ<XS:019',
              #
              'WPLGR01':'PF/WA-WPLGR<XS:001',
              'WPLGR02':'PF/WA-WPLGR<XS:002',
              'WPLGR03':'PF/WA-WPLGR<XS:003',
              'WPLGR04':'PF/WA-WPLGR<XS:004',
              'WPLGR05':'PF/WA-WPLGR<XS:005',
              'WPLGR06':'PF/WA-WPLGR<XS:006',
              'WPLGR07':'PF/WA-WPLGR<XS:007',
              'WPLGR08':'PF/WA-WPLGR<XS:008',
              'WPLGR09':'PF/WA-WPLGR<XS:009',
              'WPLGR10':'PF/WA-WPLGR<XS:010',
              'WPLGR11':'PF/WA-WPLGR<XS:011',
              'WPLGR12':'PF/WA-WPLGR<XS:012',
              'WPLGR13':'PF/WA-WPLGR<XS:013',
              'WPLGR14':'PF/WA-WPLGR<XS:014',
              'WPLGR15':'PF/WA-WPLGR<XS:015',
              'WPLGR16':'PF/WA-WPLGR<XS:016',
              'WPLGR17':'PF/WA-WPLGR<XS:017',
              'WPLGR18':'PF/WA-WPLGR<XS:018',
              'WPLGR19':'PF/WA-WPLGR<XS:019',
              'WPLGR20':'PF/WA-WPLGR<XS:020',
              'WPLGR21':'PF/WA-WPLGR<XS:021',
              'WPLGR22':'PF/WA-WPLGR<XS:022',
              'WPLGR23':'PF/WA-WPLGR<XS:023',
              'WPLGR24':'PF/WA-WPLGR<XS:024',
              'WPLGR25':'PF/WA-WPLGR<XS:025',
              #
              'WPLGZ01':'PF/WA-WPLGZ<XS:001',
              'WPLGZ02':'PF/WA-WPLGZ<XS:002',
              'WPLGZ03':'PF/WA-WPLGZ<XS:003',
              'WPLGZ04':'PF/WA-WPLGZ<XS:004',
              'WPLGZ05':'PF/WA-WPLGZ<XS:005',
              'WPLGZ06':'PF/WA-WPLGZ<XS:006',
              'WPLGZ07':'PF/WA-WPLGZ<XS:007',
              'WPLGZ08':'PF/WA-WPLGZ<XS:008',
              'WPLGZ09':'PF/WA-WPLGZ<XS:009',
              'WPLGZ10':'PF/WA-WPLGZ<XS:010',
              'WPLGZ11':'PF/WA-WPLGZ<XS:011',
              'WPLGZ12':'PF/WA-WPLGZ<XS:012',
              'WPLGZ13':'PF/WA-WPLGZ<XS:013',
              'WPLGZ14':'PF/WA-WPLGZ<XS:014',
              'WPLGZ15':'PF/WA-WPLGZ<XS:015',
              'WPLGZ16':'PF/WA-WPLGZ<XS:016',
              'WPLGZ17':'PF/WA-WPLGZ<XS:017',
              'WPLGZ18':'PF/WA-WPLGZ<XS:018',
              'WPLGZ19':'PF/WA-WPLGZ<XS:019',
              'WPLGZ20':'PF/WA-WPLGZ<XS:020',
              'WPLGZ21':'PF/WA-WPLGZ<XS:021',
              'WPLGZ22':'PF/WA-WPLGZ<XS:022',
              'WPLGZ23':'PF/WA-WPLGZ<XS:023',
              'WPLGZ24':'PF/WA-WPLGZ<XS:024',
              'WPLGZ25':'PF/WA-WPLGZ<XS:025',
              #
              'UDPGR01':'PF/WA-UDPGR<XS:001',
              'UDPGR02':'PF/WA-UDPGR<XS:002',
              'UDPGR03':'PF/WA-UDPGR<XS:003',
              'UDPGR04':'PF/WA-UDPGR<XS:004',
              'UDPGR05':'PF/WA-UDPGR<XS:005',
              'UDPGR06':'PF/WA-UDPGR<XS:006',
              'UDPGR07':'PF/WA-UDPGR<XS:007',
              'UDPGR08':'PF/WA-UDPGR<XS:008',
              #
              'UDPGZ01':'PF/WA-UDPGZ<XS:001',
              'UDPGZ02':'PF/WA-UDPGZ<XS:002',
              'UDPGZ03':'PF/WA-UDPGZ<XS:003',
              'UDPGZ04':'PF/WA-UDPGZ<XS:004',
              'UDPGZ05':'PF/WA-UDPGZ<XS:005',
              'UDPGZ06':'PF/WA-UDPGZ<XS:006',
              'UDPGZ07':'PF/WA-UDPGZ<XS:007',
              'UDPGZ08':'PF/WA-UDPGZ<XS:008',
       }

    elif signalsTableName == 'signalsTable_PFC_ITF':
           sigTable = {
              # keys  Values
           'IPRIM'  :  'PF/SC-IP1<MS'  ,  #       'P1 Ext curr'
           'IP4T'   :  'PF/SC-IP4<MS'  ,  #       'P4 total curr'
           'IP4IM'  :  'PF/SC-IIM<MS'  ,  #       'P4 Imb curr'
           'IPFX'   :  'PF/SC-IFX<MS'  ,  #       'PFX curr'
           'ISHP'   :  'PF/SC-ISH<MS'  ,  #       'Shaping curr'
           'ID1'    :  'PF/SC-ID1<MS'  ,  #       'D1 curr'
           'ID2'    :  'PF/SC-ID2<MS'  ,  #       'D2 curr'
           'ID3'    :  'PF/SC-ID3<MS'  ,  #       'D3 curr'
           'ID4'    :  'PF/SC-ID4<MS'  ,  #       'D4 curr'
           'ITF'    :  'PF/SC-ITF<MS'  ,  #
           'ISU'    :  'PF/SC-ISU<MS'  ,  #
           'IFRFA'  :  'PF/V5-IFRFA<ADC', #  ERFA current from V5
           #'IP'     :  'PF/SC-IP<MS'   ,  #       'Plasma curr'
                  #
            # ODD and EVEN TF currents KC1D
           'TFODD_KC1D' :'DA/C2-TFCI1B',   # TF current odd  # use for LOCA use after #67531
           'TFEVN_KC1D' :'DA/C2-TFCI2B',   # TF current even # use for LOCA use after #67531
           #'TFEVN' :'DA/C2-ITFEVN',   # TF current odd # use before  #67531
           #'TFODD' :'DA/C2-ITFODD',   # TF current even # use before  #67531
           #'ITFB'  :'DA/C2-ITFB'  # TF current = [C2-ITFODD + C2-ITFEVN]/2
                  # KC1Z
           #'TFEVN' :'DA/C2Z-ITFEVN',   # TF current odd
           #'TFODD' :'DA/C2Z-ITFODD',   # TF current even
           'TFODD_KC1Z' :'DA/C2Z-TFCI1',   # TF current odd
           'TFEVN_KC1Z' :'DA/C2Z-TFCI2',   # TF current even
                  #
           'TFODD_KC1E' :'DA/C2E-TFCI1',   # TF current odd # use after #69199
           'TFEVN_KC1E' :'DA/C2E-TFCI2',   # TF current even  # use after #69199
       }
    elif  signalsTableName == 'signalsTable_MHD_KC1M':
          sigTable = {
             # keys values
             # 'H301' : 'DA/C1M-H301' ,   # HRCA: High Resolution Coils Array
             # 'H302' : 'DA/C1M-H302' ,   # HRCA: High Resolution Coils Array
             # 'H303' : 'DA/C1M-H303' ,   # HRCA: High Resolution Coils Array
             # 'H304' : 'DA/C1M-H304' ,   # HRCA: High Resolution Coils Array
             # 'H305' : 'DA/C1M-H305' ,   # HRCA: High Resolution Coils Array
             # 'H306' : 'DA/C1M-H306' ,   # HRCA: High Resolution Coils Array
             # 'H307' : 'DA/C1M-H307' ,   # HRCA: High Resolution Coils Array
             #
             'I801' : 'DA/C1M-I801' ,   # IC: Internal COils
             'I802' : 'DA/C1M-I802' ,   # IC: Internal COils
             'I803' : 'DA/C1M-I803' ,   # IC: Internal COils
             #
             # 'P402'  : 'DA/C1M-P402'  , # OPLTC: Outer Poloidal Limiter Tangential Coils
             # 'P802'  : 'DA/C1M-P802'  , # OPLTC: Outer Poloidal Limiter Tangential Coils
             # #
             # 'PP401' : 'DA/C1M-PP401' , # OPLTC: Outer Poloidal Limiter Tangential Coils
             # 'PP402' : 'DA/C1M-PP402' , # OPLTC: Outer Poloidal Limiter Tangential Coils
             # 'PP403' : 'DA/C1M-PP403' , # OPLTC: Outer Poloidal Limiter Tangential Coils
             # 'PP404' : 'DA/C1M-PP404' , # OPLTC: Outer Poloidal Limiter Tangential Coils
             # 'PP405' : 'DA/C1M-PP405' , # OPLTC: Outer Poloidal Limiter Tangential Coils
             # 'PP406' : 'DA/C1M-PP406' , # OPLTC: Outer Poloidal Limiter Tangential Coils
             # 'PP407' : 'DA/C1M-PP407' , # OPLTC: Outer Poloidal Limiter Tangential Coils
             # #
             # 'PP801' : 'DA/C1M-PP801' , # OPLTC: Outer Poloidal Limiter Tangential Coils
             # 'PP802' : 'DA/C1M-PP802' , # OPLTC: Outer Poloidal Limiter Tangential Coils
             # 'PP803' : 'DA/C1M-PP803' , # OPLTC: Outer Poloidal Limiter Tangential Coils
             # 'PP804' : 'DA/C1M-PP804' , # OPLTC: Outer Poloidal Limiter Tangential Coils
             # 'PP805' : 'DA/C1M-PP805' , # OPLTC: Outer Poloidal Limiter Tangential Coils
             # 'PP806' : 'DA/C1M-PP806' , # OPLTC: Outer Poloidal Limiter Tangential Coils
             # 'PP807' : 'DA/C1M-PP807' , # OPLTC: Outer Poloidal Limiter Tangential Coils
             # #
             # 'T001' : 'DA/C1M- T001' ,  # Toroidal Coil Array
             # 'T002' : 'DA/C1M- T002' ,  # Toroidal Coil Array
             # 'T006' : 'DA/C1M- T006' ,  # Toroidal Coil Array
             # 'T007' : 'DA/C1M- T007' ,  # Toroidal Coil Array
             # 'T008' : 'DA/C1M- T008' ,  # Toroidal Coil Array
             # 'T009' : 'DA/C1M- T009' ,  # Toroidal Coil Array

             }


    return sigTable
