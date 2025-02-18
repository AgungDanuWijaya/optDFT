import torch
import numpy as np
from snn_3h import SNN
delta_kali=1e-5
scaling_factor0 = 1.0
scaling_factor1 = 1.0
scaling_factor2 = 1.0
scaling_factor3 = 0.001
input_dim = 3
output_dim = 1
depth = 4
lamda = 1e-5
beta = 1.5
use_cuda = False
device = torch.device("cuda" if use_cuda else "cpu")
hidden = [20, 20, 20]

s_nn = SNN(input_dim, output_dim, hidden, lamda, beta, use_cuda)
snn_para = s_nn.state_dict()
s_nn.to(device)

w = [-2.1328573,
         -13.64804,
         -3.0674002,
         -9.2548177,
         -12.560699,
         -4.8521864,
         -11.713733,
         -14.134225,
         -6.7884295,
         -9.8616165,
         -14.481015,
         -8.7588226,
         -10.284158,
         -0.4447621,
         -2.4351444,
         -12.4352,
         -4.2380262,
         -13.075878,
         -17.140177,
         -7.0769237,
         -6.0968678,
         -7.7292287,
         -10.567294,
         -12.000828,
         -15.202197,
         -7.0094979,
         -13.732399,
         -5.5572492,
         -11.951649,
         -14.069799,
         -3.573301,
         -10.441043,
         -0.8925175,
         -14.111414,
         -1.2772868,
         -3.6795628,
         -4.4406993,
         -3.9796117,
         -8.8798432,
         -8.1773436,
         -11.108225,
         -16.012742,
         -14.224691,
         -18.771149,
         -1.1417629,
         -4.2289558,
         -12.463067,
         -3.6344435,
         -1.9604737,
         -6.6262516,
         -10.964811,
         -10.679153,
         -3.6971526,
         -1.1883116,
         -11.743236,
         -5.2566424,
         -15.551964,
         -0.2113122,
         -15.297908,
         -6.4764345,
         -5.4781232,
         -15.589385,
         -6.6032052,
         -4.6288022,
         -6.9937693,
         -9.5575387,
         -13.48341,
         -10.222414,
         -18.366581,
         -3.8445946,
         -8.9801056,
         -9.6484784,
         -15.892782,
         -16.032551,
         -15.108185,
         -11.183047,
         -19.866151,
         -16.776351,
         -14.650216,
         -10.483571,
         -14.443778,
         -8.5295212,
         -6.651221,
         -16.750202,
         -16.676288,
         -9.801455,
         -18.629458,
         -16.387129,
         -1.9173035,
         -14.725039,
         -12.931452,
         -14.46964,
         -17.980519,
         -8.2658556,
         -5.821139,
         -8.4876956,
         -10.467695,
         -5.9582173,
         -4.6032163,
         -17.384779,
         -13.183948,
         -12.889556,
         -13.407535,
         -8.6139887,
         -10.7649,
         -6.5163267,
         -13.923666,
         -16.261353,
         -0.9847053,
         -4.3670573,
         -11.897016,
         -13.819115,
         -9.0250791,
         -17.454934,
         -15.17966,
         -5.1787819,
         -10.85531,
         -14.865905,
         -16.136548,
         -9.6455718,
         -13.831953,
         -15.132113,
         -10.859816,
         -7.5145676,
         -5.6322964,
         -11.540135,
         -17.1053,
         -17.328882,
         -12.105428,
         -10.507031,
         -10.668494,
         -11.370147,
         -7.3276333,
         -15.28088,
         -4.7789767,
         -1.0482065,
         -0.0786481,
         -8.2255391,
         -7.3238948,
         -5.1596514,
         -1.4517349,
         -16.678369,
         -11.39031,
         -4.2421385,
         -10.48516,
         -15.262527,
         -4.9430157,
         -18.173936,
         -3.9587173,
         -16.130772,
         -2.0040553,
         -8.2369356,
         -11.536223,
         -4.5866867,
         -7.944615,
         -18.711645,
         -18.681383,
         -19.314839,
         -16.113857,
         -16.436657,
         -2.5415958,
         -6.6376236,
         -3.2606728,
         -1.2421438,
         -5.6000318,
         -18.179137,
         -6.0865162,
         -0.4824433,
         -7.5231279,
         -18.426028,
         -8.4775909,
         -17.304764,
         -7.1997801,
         -2.4802945,
         -5.9725127,
         -2.3924962,
         -4.1974153,
         -10.59147,
         -7.4115966,
         -13.091909,
         -19.398047,
         -13.559961,
         -10.022977,
         -8.3900285,
         -0.2509732,
         -13.140675,
         -7.8473162,
         -7.0352544,
         -6.6206063,
         -17.118971,
         -19.274263,
         -5.3565042,
         -11.359686,
         -6.1003248,
         -9.938983,
         -9.1756668,
         -7.5504904,
         -11.450468,
         -7.65211,
         -8.2321611,
         -14.235256,
         -1.0515176,
         -10.37159,
         -6.7721247,
         -6.2286078,
         -0.1776375,
         -2.1253491,
         -2.6807181,
         -4.7610247,
         -2.1590449,
         -8.5994133,
         -12.788389,
         -14.327341,
         -15.441834,
         -11.372601,
         -1.7784884,
         -8.2548717,
         -16.719036,
         -4.6719822,
         -12.69598,
         -13.462027,
         -14.431791,
         -15.544724,
         -4.71751,
         -5.2247028,
         -15.243983,
         -5.5618622,
         -4.1334662,
         -14.973222,
         -15.035981,
         -16.41422,
         -10.23425,
         -6.0003877,
         -11.271424,
         -10.819375,
         -10.46404,
         -6.9109313,
         -3.743419,
         -17.34868,
         -0.7446179,
         -11.314005,
         -4.9810816,
         -19.547382,
         -12.760299,
         -1.0285269,
         -2.643138,
         -16.337995,
         -13.468948,
         -3.0918129,
         -11.475513,
         -17.232584,
         -7.7269214,
         -6.1166397,
         -16.043435,
         -18.929185,
         -6.6135854,
         -13.327189,
         -17.415952,
         -4.872706,
         -10.700462,
         -8.1660439,
         -8.9685591,
         -8.1750188,
         -14.398494,
         -9.5261031,
         -3.3694572,
         -17.356874,
         -7.1748555,
         -15.128814,
         -18.360285,
         -16.998118,
         -9.9365582,
         -7.8412136,
         -4.7228404,
         -9.1338787,
         -11.449611,
         -1.3739705,
         -17.225285,
         -10.37679,
         -12.445729,
         -2.6363571,
         -10.842232,
         -15.722082,
         -2.1740118,
         -9.9722552,
         -6.2020697,
         -5.2094911,
         -4.867587,
         -9.0718946,
         -11.016324,
         -0.0298641,
         -8.9443459,
         -15.267303,
         -8.8729174,
         -17.891154,
         -18.497578,
         -7.7059227,
         -2.6836585,
         -18.74507,
         -5.7115716,
         -10.687192,
         -16.094234,
         -19.220226,
         -14.25512,
         -13.133825,
         -18.353011,
         -16.809326,
         -6.1926105,
         -0.9431909,
         -9.2968941,
         -3.3622195,
         -11.05326,
         -5.1496984,
         -2.759051,
         -14.965786,
         -12.765351,
         -8.2713234,
         -12.41267,
         -5.3619714,
         -0.8175613,
         -4.2817586,
         -17.904595,
         -1.3788696,
         -0.8042198,
         -16.848284,
         -15.655428,
         -12.507432,
         -5.7986662,
         -8.2273575,
         -7.7363027,
         -5.3294609,
         -10.935696,
         -13.575112,
         -16.894024,
         -9.0244302,
         -4.3543803,
         -3.0780236,
         -19.920602,
         -3.3992373,
         -17.869656,
         -17.835608,
         -15.253166,
         -6.0666383,
         -2.9863441,
         -5.4371434,
         -16.440393,
         -13.044876,
         -8.5511006,
         -3.6726712,
         -9.7667016,
         -4.2051977,
         -12.509363,
         -3.9064138,
         -6.2440855,
         -14.02024,
         -3.068802,
         -16.502385,
         -9.2274968,
         -11.1408,
         -14.53755,
         -12.827026,
         -7.430808,
         -10.505493,
         -7.069445,
         -11.192157,
         -12.530699,
         -4.3510907,
         -1.7861007,
         -4.2255234,
         -6.2570037,
         -12.280383,
         -12.273452,
         -0.2350811,
         -19.605339,
         -17.391262,
         -11.734947,
         -6.936443,
         -0.0460641,
         -14.600342,
         -4.6649211,
         -10.70601,
         -10.751501,
         -8.7460995,
         -10.979815,
         -10.218928,
         -12.886109,
         -3.919462,
         -10.216879,
         -17.71395,
         -7.1119799,
         -0.6021096,
         -19.940277,
         -18.656972,
         -11.565325,
         -2.5471455,
         -13.925669,
         -14.486815,
         -11.805643,
         -13.658657,
         -16.535232,
         -7.4167554,
         -7.8039307,
         -8.61609,
         -17.435537,
         -6.5254388,
         -9.2516586,
         -10.825905,
         -4.6527564,
         -10.374558,
         -9.8261516,
         -13.21859,
         -10.837566,
         -9.8820698,
         -7.3749501,
         -5.5467056,
         -8.3195035,
         -14.370279,
         -13.784174,
         -3.8349099,
         -14.047462,
         -1.4476083,
         -13.077992,
         -11.936717,
         -15.362812,
         -11.355194,
         -14.305214,
         -4.4195976,
         -1.0364926,
         -18.331401,
         -12.315122,
         -16.3552,
         -19.342655,
         -8.8114047,
         -12.293895,
         -6.5602703,
         -0.3742923,
         -9.9302617,
         -11.9033,
         -7.1806266,
         -19.424484,
         -18.395584,
         -2.0263374,
         -13.411555,
         -5.6695225,
         -5.2604575,
         -6.6753704,
         -3.4419463,
         -9.5298015,
         -8.396589,
         -12.006341,
         -6.6378467,
         -7.8369704,
         -19.707614,
         -4.2130508,
         -10.937477,
         -5.5968215,
         -3.1132632,
         -5.0462072,
         -3.6690847,
         -12.706071,
         -17.239657,
         -12.558501,
         -14.594982,
         -7.2391344,
         -13.31504,
         -15.497112,
         -7.1776037,
         -6.511519,
         -13.994861,
         -0.5941158,
         -7.1036394,
         -12.923479,
         -8.6057527,
         -4.6001159,
         -15.832102,
         -18.120533,
         -19.369003,
         -3.0887038,
         -9.4255072,
         -11.278399,
         -2.6960906,
         -0.4318966,
         -8.1022292,
         -8.8228637,
         -12.250758,
         -13.7547,
         -2.6105538,
         -16.221623,
         -14.616832,
         -9.3760867,
         -7.5700824,
         -9.549542,
         -10.881044,
         -7.8567246,
         -0.9399677,
         -19.538286,
         -9.7011389,
         -8.6718663,
         -13.492503,
         -17.43701,
         -2.9598352,
         -17.676045,
         -6.1575117,
         -12.376342,
         -12.644111,
         -7.7971518,
         -13.106362,
         -7.424083,
         -13.091832,
         -18.246902,
         -16.544079,
         -16.965723,
         -11.488934,
         -16.441064,
         -8.7438426,
         -10.03965,
         -2.7734764,
         -8.6720431,
         -2.924842,
         -17.911977,
         -9.7755202,
         -8.9093945,
         -13.541236,
         -16.48681,
         -2.0341255,
         -7.9336768,
         -14.534348,
         -1.3563728,
         -0.179692,
         -14.433782,
         -5.9502204,
         -11.757089,
         -12.205012,
         -7.0179764,
         -14.047086,
         -4.5653871,
         -14.540562,
         -12.483745,
         -12.601212,
         -3.6838786,
         -18.874658,
         -4.5840286,
         -17.419459,
         -0.7321293,
         -15.644172,
         -14.13916,
         -13.694276,
         -19.059904,
         -13.950495,
         -8.142061,
         -11.203344,
         -15.040586,
         -14.366682,
         -6.6406087,
         -7.0886373,
         -0.8901883,
         -6.809926,
         -18.027175,
         -5.7918674,
         -16.92769,
         -13.949236,
         -11.693489,
         -3.9645787,
         -3.1131301,
         -18.098252,
         -7.5863101,
         -10.611802,
         -11.708615,
         -18.117716,
         -12.356654,
         -10.539049,
         -10.767837,
         -6.6357864,
         -5.5207045,
         -13.536224,
         -2.4956794,
         -3.9114116,
         -10.282461,
         -0.7390568,
         -13.797358,
         -10.093881,
         -0.9255997,
         -19.196574,
         -4.8071369,
         -18.421202,
         -5.6270138,
         -10.708146,
         -3.005016,
         -12.186737,
         -13.011491,
         -9.8151184,
         -11.967556,
         -1.8250247,
         -0.370521,
         -6.628806,
         -10.03185,
         -0.3188915,
         -17.278095,
         -11.084931,
         -6.8121491,
         -8.3141071,
         -1.918386,
         -13.05844,
         -12.554581,
         -10.065154,
         -6.8955764,
         -12.746415,
         -8.0196574,
         -8.0674141,
         -10.245575,
         -15.960417,
         -7.5279074,
         -8.0988688,
         -5.2863936,
         -3.520181,
         -2.5930376,
         -16.991078,
         -19.277667,
         -4.8052188,
         -19.644164,
         -13.707356,
         -11.692769,
         -4.3171699,
         -12.728831,
         -16.345176,
         -15.113228,
         -8.850179,
         -19.407037,
         -17.628028,
         -16.164188,
         -5.7405849,
         -1.6010488,
         -5.5947004,
         -17.283279,
         -12.364105,
         -11.921974,
         -8.7955207,
         -12.931381,
         -9.5541789,
         -15.076153,
         -3.3668899,
         -11.671124,
         -9.2877006,
         -8.6865063,
         -6.6335785,
         -13.016276,
         -14.834148,
         -6.1542153,
         -8.8929327,
         -15.027745,
         -11.189592,
         -14.030844,
         -14.657167,
         -12.620155,
         -5.3884529,
         -7.520683,
         -13.191367,
         -3.3355417,
         -8.2958799,
         -11.232501,
         -12.576297,
         -17.813827,
         -9.8712676,
         -17.532166,
         -15.23434,
         -7.6528844,
         -11.054796,
         -2.6115866,
         -7.1969057,
         -4.4257915,
         -11.863299,
         -3.5837654,
         -12.621899,
         -11.137503,
         -4.4636342,
         -10.193528,
         -8.1566253,
         -16.505378,
         -13.083803,
         -6.3967749,
         -19.967996,
         -19.320523,
         -17.867184,
         -16.866824,
         -19.309628,
         -4.2458024,
         -12.028128,
         -14.541054,
         -8.912936,
         -19.949738,
         -13.613974,
         -16.278621,
         -13.805033,
         -11.523756,
         -2.4223771,
         -5.1690501,
         -11.733953,
         -14.329251,
         -14.228764,
         -19.565733,
         -8.4662797,
         -15.843747,
         -11.788142,
         -6.5101783,
         -18.07056,
         -9.5314467,
         -11.577152,
         -3.7039884,
         -9.7572201,
         -17.164402,
         -2.4324923,
         -6.436946,
         -16.735678,
         -9.3962339,
         -8.2372411,
         -11.777945,
         -11.052199,
         -6.1372633,
         -4.9698493,
         -6.2996285,
         -4.5634798,
         -18.262357,
         -3.5649802,
         -10.901573,
         -4.6911664,
         -4.9362356,
         -11.327091,
         -0.0395041,
         -18.472324,
         -5.7944128,
         -13.988113,
         -11.27597,
         -6.8571597,
         -2.6432994,
         -6.241738,
         -3.0336707,
         -5.4477484,
         -8.487368,
         -10.992923,
         -12.2277,
         -1.7034265,
         -7.5616737,
         -11.821328,
         -6.307884,
         -14.701752,
         -7.6643318,
         -10.320683,
         -11.476425,
         -18.792731,
         -11.215081,
         -5.3580788,
         -15.29384,
         -6.3878184,
         -6.0791564,
         -16.665342,
         -10.517414,
         -9.3903305,
         -12.650287,
         -1.8223418,
         -12.185541,
         -14.927846,
         -11.250428,
         -7.9552502,
         -15.271315,
         -18.260393,
         -9.0274939,
         -9.0937371,
         -16.971188,
         -15.000773,
         -7.7157341,
         -10.268019,
         -18.660544,
         -11.708037,
         -9.2125122,
         -5.2838327,
         -13.861213,
         -17.607341,
         -3.2938608,
         -10.140941,
         -11.15002,
         -5.857913,
         -6.868287,
         -3.1091941,
         -9.7679018,
         -11.856144,
         -0.480237,
         -0.6895447,
         -10.522723,
         -7.6390142,
         -18.50447,
         -15.560525,
         -3.4497873,
         -18.417037,
         -12.202286,
         -12.389659,
         -7.6897975,
         -9.4755903,
         -14.986319,
         -10.198203,
         -5.2780696,
         -0.8758027,
         -3.0175841,
         -6.7674516,
         -8.5154213,
         -8.5496402,
         -11.368773,
         -12.046469,
         -2.9493517,
         -7.3356606,
         -12.161837,
         -6.9932972,
         -1.4575094,
         -16.803411,
         -13.554568,
         -6.5705534,
         -10.201866,
         -11.598868,
         -15.405033,
         -18.322761,
         -17.821912,
         -17.355736,
         -11.183407,
         -15.108751,
         -9.6008773,
         -7.8418415,
         -5.2177858,
         -3.3601249,
         -6.8607963,
         -3.2236493,
         -9.3415898,
         -11.301904,
         -13.334888,
         -0.5349775,
         -2.0832381,
         -1.7346055,
         -11.773986,
         -2.6003667,
         -12.303914,
         -9.6959925,
         -12.234757,
         -13.197463,
         -8.4430735,
         -6.0674876,
         -17.64493,
         -19.703791,
         -15.187942,
         -10.534263,
         -4.5053224,
         -11.484273,
         -13.084669,
         -16.986498,
         -7.9780607,
         -14.584069,
         -0.1290807,
         -1.2967631,
         -11.063867,
         -17.367697,
         -4.3408797,
         -12.818699,
         -13.44657,
         -6.904023,
         -11.391938,
         -6.1434898,
         -8.517284,
         -1.6125312,
         -14.246912,
         -14.466388,
         -10.605768,
         -4.1775981,
         -17.559807,
         -6.4249958,
         -18.762355,
         -7.7597285,
         -2.8219435,
         -19.987832,
         -14.308066,
         -10.36655,
         -6.9520123,
         -11.94609,
         -16.59711,
         -8.3895842,
         -15.064077,
         -2.2290305,
         -0.5293448,
         -15.825642,
         -5.4292804,
         -8.7257028,
         -4.4867227,
         -0.100196,
         -3.006752,
         -17.067892,
         -11.738344,
         -13.586249,
         -4.3311202,
         -4.8481453,
         -15.123685,
         -7.2626871,
         -14.633305,
         -1.1054541,
         -15.203709,
         -8.3734243,
         -5.3830876,
         -5.1339173,
         -8.1364554,
         -14.409825,
         -7.9238078,
         -6.8521189,
         -8.0468019,
         -8.5669551,
         -4.0189374,
         -11.376052,
         -10.740492,
         -1.669223,
         -17.83699,
         -5.9285106,
         -8.7579361,
         -18.005187,
         -8.2774806,
         -8.683225,
         -6.4251612,
         -18.525034,
         -14.754472,
         -2.0176609,
         -1.8797516,
         -8.3763251,
         -14.346184,
         -9.723284,
         -13.52136,
         -17.926634,
         -8.1999113,
         -7.0308316,
         -9.4873262,
         -14.424848,
         -6.6589358,
         -16.948564,
         -10.359068,
         -15.725621,
         -18.99706,
         -12.443681,
         -6.8934659

         ]

k = 0
for i00 in range(hidden[0]):
    for j in range(input_dim):
        snn_para['model.0.weight'][i00, j] = w[k] * scaling_factor0
        k += 1
for i01 in range(hidden[0]):
    snn_para['model.0.bias'][i01] = w[k + i01] * scaling_factor0
k = hidden[0] * (input_dim + 1)
for i10 in range(hidden[1]):
    for j in range(hidden[0]):
        snn_para['model.2.weight'][i10, j] = w[k] * scaling_factor1
        k += 1
for i11 in range(hidden[1]):
    snn_para['model.2.bias'][i11] = w[k + i11] * scaling_factor1
k = hidden[0] * (input_dim + 1) + hidden[1] * (hidden[0] + 1)
for i20 in range(hidden[2]):
    for j in range(hidden[1]):
        snn_para['model.4.weight'][i20, j] = w[k] * scaling_factor2
        k += 1
for i21 in range(hidden[2]):
    snn_para['model.4.bias'][i21] = w[k + i21] * scaling_factor2
k = hidden[0] * (input_dim + 1) + hidden[1] * (hidden[0] + 1) + hidden[2] * (hidden[1] + 1)
for i3 in range(hidden[2]):
    snn_para['model.6.weight'][0, i3] = w[k + i3] * scaling_factor3
s_nn.load_state_dict(snn_para)

def lyp(rho01, rho02, gamma1, gamma2, gamma12, i):
    gamma12=abs(gamma12)
    ml_in_ = np.concatenate((rho01.reshape((-1, 1)), rho02.reshape((-1, 1)), gamma1.reshape((-1, 1)),
                             gamma2.reshape((-1, 1)), gamma12.reshape((-1, 1))), axis=1)
    ml_in = torch.Tensor(ml_in_)
    ml_in.requires_grad = True
    exc_ml_out = s_nn(ml_in, is_training_data=False)
    ml_exc = exc_ml_out.detach().numpy()

    exc = (ml_exc).reshape(-1)


    return exc


def vrho_1(a, b, gaa, gbb, gnn):
    delta1 = a * delta_kali
    exc = lyp(a, b, gaa, gbb, gnn, 1)
    exc1 = (lyp(a + delta1, b, gaa, gbb, gnn, 2))
    vrho1 = (exc1 - exc) / (delta1 + 1e-250)
    return vrho1


def vrho_2(a, b, gaa, gbb, gnn):
    delta2 = b * delta_kali
    exc = lyp(a, b, gaa, gbb, gnn, 1)
    exc2 = lyp(a, b + delta2, gaa, gbb, gnn, 3)
    vrho2 = (exc2 - exc) / (delta2 + 1e-250)
    return vrho2


def vgama_1(a, b, gaa, gbb, gnn):
    delta1 = (gaa) * delta_kali
    exc = lyp(a, b, gaa, gbb, gnn, 1)
    exc1g = lyp(a, b, gaa + delta1, gbb, gnn, 4)
    vgama1 = (exc1g - exc) / (delta1 + 1e-250)
    return vgama1


def vgama_2(a, b, gaa, gbb, gnn):
    delta2 = (gbb) * delta_kali
    exc = lyp(a, b, gaa, gbb, gnn, 1)
    exc2g = lyp(a, b, gaa, gbb + delta2, gnn, 5)
    vgama2 = (exc2g - exc) / (delta2 + 1e-250)
    return vgama2


def vgama_3(a, b, gaa, gbb, gnn):
    delta3 = (gnn) * delta_kali
    exc = lyp(a, b, gaa, gbb, gnn, 1)
    exc3g = lyp(a, b, gaa, gbb, gnn + delta3, 6)
    vgama3 = (exc3g - exc) / (delta3 + 1e-250)
    return vgama3


def dxc(a, b, gaa, gbb, gnn):
    vrho1 = vrho_1(a, b, gaa, gbb, gnn)
    vrho2 = vrho_2(a, b, gaa, gbb, gnn)

    vrhoc = [vrho1, vrho2]

    vgama1 = vgama_1(a, b, gaa, gbb, gnn)
    vgama2 = vgama_2(a, b, gaa, gbb, gnn)
    vgama3 = vgama_3(a, b, gaa, gbb, gnn)

    vgamac = [vgama1, vgama3, vgama2]

    delta1_ = (a) * delta_kali
    vrho1_ = vrho_1(a + delta1_, b, gaa, gbb, gnn)
    v2rho1 = (vrho1_ - vrho1) / (delta1_+ 1e-250)

    delta2_ = (b) * delta_kali
    vrho2_ = vrho_1(a, b + delta2_, gaa, gbb, gnn)
    v2rho21 = (vrho2_ - vrho1) / (delta2_+ 1e-250)

    delta2_ = (b) * delta_kali
    vrho2_ = vrho_2(a, b + delta2_, gaa, gbb, gnn)
    v2rho2 = (vrho2_ - vrho2) / (delta2_+ 1e-250)

    v2rhoc = [v2rho1, v2rho21, v2rho2]

    delta1 = (gaa) * delta_kali
    vgama1_ = vgama_1(a, b, gaa + delta1, gbb, gnn)
    vtautau11 = (vgama1_ - vgama1) / (delta1+ 1e-250)

    delta1 = (gaa) * delta_kali
    vgama2_ = vgama_2(a, b, gaa + delta1, gbb, gnn)
    vtautau12 = (vgama2_ - vgama2) / (delta1+ 1e-250)

    delta1 = (gaa) * delta_kali
    vgama3_ = vgama_3(a, b, gaa + delta1, gbb, gnn)
    vtautau13 = (vgama3_ - vgama3) / (delta1+ 1e-250)

    delta2 = (gbb) * delta_kali
    vgama2_ = vgama_2(a, b, gaa, gbb + delta2, gnn)
    vtautau22 = (vgama2_ - vgama2) / (delta2+ 1e-250)

    delta2 = (gbb) * delta_kali
    vgama3_ = vgama_3(a, b, gaa, gbb + delta2, gnn)
    vtautau23 = (vgama3_ - vgama3) / (delta2+ 1e-250)

    delta3 = (gnn) * delta_kali
    vgama3_ = vgama_3(a, b, gaa, gbb, gnn + delta3)
    vtautau33 = (vgama3_ - vgama3) / (delta3+ 1e-250)

    vtautauc = [vtautau11, vtautau12, vtautau13, vtautau22, vtautau23, vtautau33]

    # =================
    delta1 = (a) * delta_kali
    vgama1_ = vgama_1(a + delta1, b, gaa, gbb, gnn)
    vrhotau11 = (vgama1_ - vgama1) / (delta1+ 1e-250)

    delta1 = (a) * delta_kali
    vgama2_ = vgama_2(a + delta1, b, gaa, gbb, gnn)
    vrhotau12 = (vgama2_ - vgama2) / (delta1+ 1e-250)

    delta1 = (a) * delta_kali
    vgama3_ = vgama_3(a + delta1, b, gaa, gbb, gnn)
    vrhotau13 = (vgama3_ - vgama3) / (delta1+ 1e-250)

    delta2 = (b) * delta_kali
    vgama1_ = vgama_1(a, b + delta2, gaa, gbb, gnn)
    vrhotau21 = (vgama1_ - vgama1) / (delta2+ 1e-250)

    delta2 = (b) * delta_kali
    vgama2_ = vgama_2(a, b + delta2, gaa, gbb, gnn)
    vrhotau22 = (vgama2_ - vgama2) / (delta2+ 1e-250)

    delta2 = (b) * delta_kali
    vgama3_ = vgama_3(a, b + delta2, gaa, gbb, gnn)
    vrhotau23 = (vgama3_ - vgama3) / (delta2+ 1e-250)

    vrhotauc = [vrhotau11, vrhotau12, vrhotau13, vrhotau21, vrhotau22, vrhotau23]

    dxc = [vrhoc, vgamac, v2rhoc, vrhotauc, vtautauc]
    return dxc




def eval_xc_gga(xc_code, rho, spin, relativity=0, deriv=2, verbose=None, omega=None):
    rho1 = rho[0]
    rho2 = rho[1]

    a, dx1, dy1, dz1 = rho1[:4]
    b, dx2, dy2, dz2 = rho2[:4]
    gaa = dx1 ** 2 + dy1 ** 2 + dz1 ** 2
    gbb = dx2 ** 2 + dy2 ** 2 + dz2 ** 2
    gnn = (dx1 * dx2) + (dy1 * dy2) + (dz1 * dz2)

    exc = lyp(a, b, gaa, gbb, gnn, 1)
    dx = dxc(a, b, gaa, gbb, gnn)
    al = 1
    fxc_ = al * np.array(dx[2])
    fxc_1 = al * np.array(dx[3])
    fxc_2 = al * np.array(dx[4])
    vgamma_ = al * np.array(dx[1])
    vgamma = np.transpose(vgamma_)
    vrho_ = al * np.array(dx[0])
    vrho = np.transpose(vrho_)
    exc = al * np.array([exc] )/ (a + b + 1e-250)
    exc = np.transpose(exc)
    vxc = (vrho, vgamma, None, None)
    fxc = (np.transpose(fxc_), fxc_1.T, fxc_2.T)

    kxc = None  # 3rd order functional derivative
    return exc, vxc, fxc, kxc


