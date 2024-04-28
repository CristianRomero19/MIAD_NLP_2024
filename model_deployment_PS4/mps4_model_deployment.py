#!/usr/bin/python

import pandas as pd
import joblib
import sys
import os


def precio_carro(year, mileage, state, make, model):

    state = ' '+state

    mapeo_state={' FL': 9, ' OH': 35, ' TX': 43, ' CO': 5, ' ME': 21, ' WA': 47, ' CT': 6, ' CA': 4, ' LA': 18, ' NY': 34, ' PA': 38, ' SC': 40,
                 ' ND': 28, ' NC': 27, ' GA': 10, ' AZ': 3, ' TN': 42, ' KY': 17, ' NJ': 31, ' UT': 44, ' IA': 12, ' AL': 1, ' NE': 29, ' IL': 14,
                 ' OK': 36, ' MD': 20, ' NV': 33, ' WV': 49, ' MI': 22, ' VA': 45, ' WI': 48, ' MA': 19, ' OR': 37, ' IN': 15, ' NM': 32, ' MO': 24,
                 ' HI': 11, ' KS': 16, ' AR': 2, ' MN': 23, ' MS': 25, ' MT': 26, ' AK': 0, ' VT': 46, ' SD': 41, ' NH': 30, ' DE': 8, ' ID': 13,
                 ' RI': 39, ' WY': 50, ' DC': 7}
    
    mapeo_make= {'Jeep': 17, 'Chevrolet': 6, 'BMW': 2, 'Cadillac': 5, 'Mercedes-Benz': 24, 'Toyota': 35, 'Buick': 4, 'Dodge': 8, 'Volkswagen': 36,
                 'GMC': 12, 'Ford': 10, 'Hyundai': 14, 'Mitsubishi': 26, 'Honda': 13, 'Nissan': 27, 'Mazda': 23, 'Volvo': 37, 'Kia': 18, 'Subaru': 32,
                 'Chrysler': 7, 'INFINITI': 15, 'Land': 19, 'Porsche': 29, 'Lexus': 20, 'MINI': 22, 'Lincoln': 21, 'Audi': 1, 'Ram': 30, 'Mercury': 25,
                 'Tesla': 34, 'FIAT': 9, 'Acura': 0, 'Scion': 31, 'Pontiac': 28, 'Jaguar': 16, 'Bentley': 3, 'Suzuki': 33, 'Freightliner': 11}

    mapeo_model = {'Wrangler': 489, 'Tahoe4WD': 448, 'X5AWD': 499, 'SRXLuxury': 398, '3': 11, 'C-ClassC300': 59, 'CamryL': 87, 'TacomaPreRunner': 446,
                   'LaCrosse4dr': 272, 'ChargerSXT': 101, 'CamryLE': 88, 'Jetta': 264, 'AcadiaFWD': 38, 'EscapeSE': 169, 'SonataLimited': 419,
                   'Santa': 400, 'Outlander': 328, 'CruzeSedan': 129, 'Civic': 104, 'CorollaL': 122, '350Z2dr': 19, 'EdgeSEL': 146, 'F-1502WD': 186,
                   'FocusSE': 217, 'PatriotSport': 343, 'Accord': 40, 'MustangGT': 310, 'FusionHybrid': 231, 'ColoradoCrew': 113, 'Wrangler4WD': 491,
                   'CR-VEX-L': 66, 'CTS': 72, 'CherokeeLimited': 102, 'Yukon': 518, 'Elantra': 148, 'New': 317, 'CorollaLE': 123, 'Canyon4WD': 92,
                   'Golf': 247, 'Sonata4dr': 418, 'Elantra4dr': 149, 'PatriotLatitude': 341, 'Mazda35dr': 299, 'Tacoma2WD': 443, 'Corolla4dr': 121,
                   'Silverado': 417, 'TerrainFWD': 458, 'EscapeFWD': 165, 'Grand': 248, 'RAV4FWD': 368, 'Liberty4WD': 280, 'FocusTitanium': 220,
                   'DurangoAWD': 135, 'S60T5': 393, 'CivicLX': 107, 'MuranoAWD': 305, 'ForteEX': 224, 'TraverseAWD': 469, 'CamaroConvertible': 82,
                   'Sportage2WD': 429, 'Pathfinder4WD': 337, 'Highlander4dr': 251, 'WRXSTI': 488, 'Ram': 378, 'F-150XLT': 197, 'SiennaXLE': 415,
                   'LaCrosseFWD': 274, 'RogueFWD': 389, 'CamaroCoupe': 83, 'JourneySXT': 267, 'AccordEX-L': 42, 'Escape4WD': 163, 'OptimaEX': 323,
                   'FusionSE': 233, '5': 27, 'F-150SuperCrew': 195, '200Limited': 6, 'Malibu': 291, 'CompassSport': 118, 'G37': 236, 'CanyonCrew': 93,
                   'Malibu1LT': 292, 'MustangPremium': 311, 'MustangBase': 308, 'Sierra': 416, 'FlexLimited': 211, 'Tahoe2WD': 447, 'Transit': 468,
                   'Outback2.5i': 326, 'TucsonLimited': 473, 'Rover': 390, 'CayenneAWD': 95, 'MalibuLT': 295, 'TucsonFWD': 472, 'F-150FX2': 188,
                   'Camaro2dr': 81, 'Colorado4WD': 112, 'SonataSE': 420, 'ESES': 141, 'EnclavePremium': 155, 'CR-VEX': 65, 'F-150STX': 194,
                   'Impreza': 261, 'EquinoxFWD': 158, 'Cooper': 120, 'Super': 438, 'Passat4dr': 335, '911': 31, 'CivicEX': 105, 'CamrySE': 89,
                   'Highlander4WD': 250, 'Corvette2dr': 125, '200S': 7, 'PilotLX': 348, 'SorentoEX': 424, 'RioLX': 388, 'ExplorerXLT': 184,
                   'CorvetteCoupe': 127, 'EnclaveLeather': 154, 'Avalanche4WD': 50, 'TacomaBase': 445, 'Versa5dr': 483, 'MKXFWD': 288,
                   'SL-ClassSL500': 396, 'VeracruzFWD': 481, 'CorollaS': 124, 'PriusTwo': 358, 'CR-V2WD': 63, 'Lucerne4dr': 283, '4Runner4dr': 22,
                   'PilotTouring': 350, 'CR-VLX': 67, 'CompassLatitude': 116, 'Altima4dr': 46, 'OptimaLX': 324, 'Focus5dr': 215, 'Charger4dr': 99,
                   'AcadiaAWD': 37, 'JourneyFWD': 266, '7': 30, 'RX': 375, 'MalibuLS': 294, 'LSLS': 269, 'SportageLX': 432, 'Yukon4WD': 520,
                   'SorentoLX': 425, 'TiguanSEL': 462, 'Camry4dr': 85, 'F-1504WD': 187, 'PriusBase': 353, 'AccordLX': 43, 'Q7quattro': 360, 'ExplorerLimited': 183, '4RunnerSR5': 25, 'OdysseyEX-L': 319, 'C-ClassC': 58, 'CX-9FWD': 77, 'JourneyAWD': 265, 'Sorento2WD': 423, 'F-250Lariat': 199, 'Prius': 351, 'TahoeLT': 451, '25004WD': 10, 'Escalade4dr': 161, 'GTI4dr': 242, '4RunnerRWD': 24, 'FX35AWD': 207, 'XC90T6': 507, 'Taurus4dr': 452, 'AvalonXLE': 54, '300300S': 13, 'G35': 235, 'F-150Platinum': 193, 'TerrainAWD': 457, 'GXGX': 244, 'MKXAWD': 287, 'Town': 467, 'CamryXLE': 90, 'VeracruzAWD': 480, 'FusionS': 232, 'Challenger2dr': 97, 'Tundra': 474, 'Navigator4WD': 315, 'Legacy3.6R': 279, 'GS': 239, 'E-ClassE350': 139, 'Suburban2WD': 435, 'A44dr': 34, 'RegalTurbo': 385, 'Outback3.6R': 327, '4Runner4WD': 21, 'Legacy2.5i': 278, '1': 0, 'Yukon2WD': 519, 'Explorer': 177, 'PilotEX-L': 347, '200LX': 5, 'M-ClassML350': 284, 'RAV4XLE': 372, 'WranglerSport': 494, 'Model': 302, 'FJ': 206, 'Titan': 463, 'Titan4WD': 465, 'FlexSEL': 213, 'OdysseyTouring': 321, 'SorentoSX': 426, 'RAV4Base': 367, 'OdysseyEX': 318, 'Explorer4WD': 178, 'Mustang2dr': 307, 'EdgeLimited': 144, 'FusionSEL': 234, 'Yukon4dr': 521, 'Touareg4dr': 466, 'Matrix5dr': 296, 'CTCT': 71, 'CherokeeSport': 103, '6': 29, 'Maxima4dr': 297, 'Frontier4WD': 229, 'PriusThree': 357, 'F-350XL': 204, '500Pop': 28, 'RDXAWD': 373, 'Tacoma4WD': 444, 'Optima4dr': 322, 'Q5quattro': 359, 'X3xDrive28i': 498, 'RDXFWD': 374, 'X5xDrive35i': 500, 'Malibu4dr': 293, 'ExpeditionXLT': 176, 'Ranger2WD': 379, 'Patriot4WD': 340, 'Quest4dr': 363, 'TaurusSE': 454, 'PathfinderS': 338, 'Murano2WD': 304, 'LS': 268, 'SiennaLimited': 413, 'ES': 140, 'SiennaLE': 412, 'F-150Lariat': 191, 'Titan2WD': 464, 'Durango2WD': 133, 'Tahoe4dr': 449, 'Focus4dr': 214, 'YarisBase': 516, 'TaurusLimited': 453, 'RAV44WD': 365, 'C-Class4dr': 57, 'Soul+': 427, 'TundraBase': 477, 'Expedition': 172, 'ImpalaLT': 260, 'SedonaLX': 404, 'Sequoia4WD': 406, 'ElantraLimited': 150, '15002WD': 1, 'Suburban4WD': 436, 'FiestaSE': 209, '15004WD': 2, 'TundraSR5': 479, 'Camry': 84, 'RAV4Limited': 370, 'RangerSuperCab': 381, 'MDXAWD': 286, 'RAV4LE': 369, 'ChallengerR/T': 98, 'FlexSE': 212, 'ForteLX': 225, 'TraverseFWD': 470, 'LibertySport': 282, 'ISIS': 257, 'Impala4dr': 258, 'Tundra4WD': 476, 'F-250XLT': 201, 'RXRX': 377, 'Armada2WD': 47, 'Frontier': 227, 'WranglerRubicon': 492, 'EquinoxAWD': 157, 'PilotEX': 346, 'TiguanS': 460, 'EscaladeAWD': 162, 'DTS4dr': 130, 'Pilot2WD': 344, 'Express': 185, 'PacificaLimited': 332, 'CanyonExtended': 94, 'MX5': 290, 'EscapeS': 168, 'IS': 256, 'C-ClassC350': 60, 'Compass4WD': 115, 'SportageEX': 431, 'Legacy': 277, 'E-ClassE': 137, 'Dakota4WD': 132, '300300C': 12, 'Forte': 223, 'SportageAWD': 430, 'TaurusSEL': 455, 'Xterra4WD': 512, 'GSGS': 240, 'Explorer4dr': 179, 'F-150XL': 196, 'SportageSX': 433, 'xB5dr': 523, 'TundraLimited': 478, 'CruzeLT': 128, 'Wrangler2dr': 490, 'HighlanderFWD': 253, 'Sprinter': 434, 'Highlander': 249, 'Prius5dr': 352, 'CX-9Grand': 78, 'CTS4dr': 74, 'Econoline': 143, 'AccordEX': 41, 'RAV4Sport': 371, '35004WD': 18, 'ChargerSE': 100, 'OdysseyLX': 320, 'TucsonAWD': 471, 'CX-7FWD': 75, 'AccordLX-S': 44, 'Navigator4dr': 316, 'EscapeXLT': 170, 'TiguanSE': 461, 'Cayman2dr': 96, 'TaurusSHO': 456, 'F-150FX4': 189, 'Ranger4WD': 380, 'OptimaSX': 325, 'SequoiaSR5': 410, 'G64dr': 237, 'HighlanderLimited': 254, 'ExplorerFWD': 182, 'F-350King': 202, 'PriusFive': 354, 'Yaris4dr': 515, 'PatriotLimited': 342, 'Lancer4dr': 275, 'HighlanderSE': 255, 'CompassLimited': 117, 'S2000Manual': 391, 'F-250King': 198, 'Forester2.5X': 221, 'Fusion4dr': 230, 'Frontier2WD': 228, 'FocusST': 219, 'Pathfinder2WD': 336, 'Sentra4dr': 405, 'XF4dr': 508, 'F-250XL': 200, 'PacificaTouring': 333, 'MustangDeluxe': 309, 'Caliber4dr': 80, 'GTI2dr': 241, 'Mazda34dr': 298, 'FocusS': 216, 'Sienna5dr': 411, 'CR-V4WD': 64, 'CX-9Touring': 79, 'Mazda64dr': 300, 'Forester4dr': 222, '1500Tradesman': 4, 'MDX4WD': 285, 'Escalade': 159, 'TL4dr': 439, 'CX-9AWD': 76, 'Canyon2WD': 91, 'A64dr': 35, 'A8': 36, 'Armada4WD': 48, 'Impreza2.0i': 262, 'GX': 243, 'QX564WD': 362, 'CC4dr': 62, 'MKZ4dr': 289, 'Yaris': 514, 'FitSport': 210, 'Regal4dr': 382, 'Tundra2WD': 475, 'X3AWD': 497, 'SonicSedan': 422, 'Cobalt4dr': 110, 'RidgelineRTL': 386, 'CivicSi': 108, 'AvalonLimited': 52, 'XC90FWD': 506, 'Outlander2WD': 329, 'RAV44dr': 366, 'ColoradoExtended': 114, 'ExpeditionLimited': 175, '3004dr': 14, '200Touring': 8, 'SC': 395, 'X1xDrive28i': 496, 'SonicHatch': 421, 'GLI4dr': 238, 'PilotSE': 349, 'Savana': 401, 'RegalPremium': 384, 'CR-VSE': 68, 'RegalGS': 383, 'XC90AWD': 505, 'EdgeSport': 147, 'PriusFour': 355, 'SiennaSE': 414, '1500Laramie': 3, '300Base': 15, 'Pilot4WD': 345, 'A34dr': 33, 'HighlanderBase': 252, 'Expedition4WD': 174, 'STS4dr': 399, 'SoulBase': 428, 'Xterra2WD': 511, 'CT': 70, 'tC2dr': 522, 'Tiguan2WD': 459, 'CR-ZEX': 69, 'MustangShelby': 312, 'C702dr': 61, 'WranglerX': 495, 'WranglerSahara': 493, 'DurangoSXT': 136, 'Sequoia4dr': 407, 'Outlander4WD': 330, 'Expedition2WD': 173, 'Navigator': 313, '9112dr': 32, 'Vibe4dr': 484, 'F-150King': 190, '300Limited': 16, 'XC60T6': 503, 'CivicEX-L': 106, 'Avalanche2WD': 49, 'F-350XLT': 205, 'ExplorerBase': 180, 'MuranoS': 306, 'LXLX': 271, 'EdgeSE': 145, 'ImpalaLS': 259, 'Land': 276, 'E-ClassE320': 138, 'Milan4dr': 301, 'Boxster2dr': 56, 'RAV4': 364, 'Eos2dr': 156, 'SedonaEX': 403, 'xD5dr': 524, 'Colorado2WD': 111, 'Monte': 303, 'Escape4dr': 164, 'LX': 270, 'FiestaS': 208, 'F-350Lariat': 203, 'Galant4dr': 245, 'TT2dr': 442, 'Xterra4dr': 513, 'SequoiaLimited': 408, '4RunnerLimited': 23, 'Genesis': 246, 'Suburban4dr': 437, 'EnclaveConvenience': 153, 'LaCrosseAWD': 273, 'Versa4dr': 482, 'Cobalt2dr': 109, 'XC60FWD': 502, 'F-150Limited': 192, 'Dakota2WD': 131, 'S44dr': 392, '4Runner2WD': 20, 'Sedona4dr': 402, 'RidgelineSport': 387, 'TSXAutomatic': 441, 'ImprezaSport': 263, 'SLK-ClassSLK350': 397, 'Accent4dr': 39, 'CorvetteConvertible': 126, 'Avalon4dr': 51, 'Passat': 334, '25002WD': 9, 'ExplorerEddie': 181, 'LibertyLimited': 281, 'CTS-V': 73, '4RunnerTrail': 26, 'Eclipse3dr': 142, 'Azera4dr': 55, 'TahoeLS': 450, 'Continental': 119, 'XJ4dr': 509, 'ForteSX': 226, 'SequoiaPlatinum': 409, 'FocusSEL': 218, 'Durango4dr': 134, 'CamryBase': 86, 'XC704dr': 504, 'S804dr': 394, 'Element4WD': 152, 'YarisLE': 517, 'WRXBase': 485, 'TLAutomatic': 440, 'AvalonTouring': 53, 'XK2dr': 510, 'PT': 331, 'PathfinderSE': 339, '300Touring': 17, 'Navigator2WD': 314, 'XC60AWD': 501, 'EscapeLimited': 167, 'WRXLimited': 486, 'AccordSE': 45, 'QX562WD': 361, 'Escalade2WD': 160, 'EscapeLImited': 166, 'PriusOne': 356, 'Element2WD': 151, 'Excursion137"': 171, 'WRXPremium': 487, 'RX-84dr': 376}

    state_cod = mapeo_state[state]
    make_cod = mapeo_make[make]
    model_cod = mapeo_model[model]

    # se deben modificar los valores de este diccionario con el acceso a los modelos de los archivos plk
    # ejemplo: clf = joblib.load(os.path.dirname(__file__) + '/phishing_clf.pkl')
    modelos = {'xgb': joblib.load(os.path.dirname(__file__) + '/xgboost_price.pkl'),
               'bgrf': joblib.load(os.path.dirname(__file__) + '/bgg_random_forest_price.pkl') }
    
    y_pred = pd.DataFrame(columns=modelos.keys())
    datos = {'Year': [year],
             'Mileage': [mileage],
             'State_Encod': [state_cod],
             'Make_Encod': [make_cod],
             'Model_Encod': [model_cod]}
    X_Test = pd.DataFrame(datos)
    
    for modelo in modelos.keys():
        y_pred[modelo] = modelos[modelo].predict(X_Test)
        price = y_pred.iloc[0].mean()

    return price


if __name__ == "__main__":
    
    if len(sys.argv) == 1:
        print('Inicio carro')
        p1 = precio_carro(2014, 31909, 'MD', 'Nissan', 'MuranoAWD')
        print(p1)
        
    else:

        url = sys.argv[1]

        p1 = precio_carro(url)
        
        print(url)
        print('Probability of Phishing: ', p1)
        