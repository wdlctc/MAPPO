import structlog
import numpy as np

SNR_THRESHOLD = 2e-8

TMP_SNR = {
    50: [
        [0.978853848032438, 0.9382560820857074, 0.9675461152915266, 1, 0.9577797655189708, 0.923695842034908, 0.9721817332681983, 1, 1, 1, 1, 1, 1, 1, 1, 0.9692891603598139, 0.9711280392502496, 1, 0.9913859319110941, 0.9513317391857605, 0.944745421612361, 0.9354440118191168, 0.9723711768049565, 1, 1, 0.9960297898858861, 1, 1, 0.9786761513251866, 0.9934806573678557, 0.9687410059512311, 1, 0.9949607924712068, 1, 0.9637525332775981, 1, 0.9688022679975166, 0.9329424733783811, 0.8887046528832709, 0.8411490477349477, 0.8414619246795281, 0.8046350416446519, 0.8148553765441571, 0.8247249646180027, 0.8304487138976068, 0.8380752427305913, 0.8506775271460228, 0.8936323938466401, 0.941270938662415],
        [0.5988508990301651, 0.630366384238783, 0.6713128541314884, 0.6661487958975103, 0.6837725852388, 0.6832115038957327, 0.7190747680560797, 0.6910910584383048, 0.7131950084878207, 0.7607215067610865, 0.7773636796990233, 0.758635459127976, 0.7796734569813274, 0.7537131364243073, 0.7237379371700599, 0.676102314337192, 0.72434002615375, 0.6873811037024214, 0.695456800163106, 0.7226723121507915, 0.6780536565823291, 0.6807101401560327, 0.6937726267606644, 0.6862530580837306, 0.6936898415102563, 0.6673718306629821, 0.7033841346215594, 0.6990343121641432, 0.6900433178862642, 0.6545087479881466, 0.6378030810326808, 0.5949066131282662, 0.5981688182675696, 0.5768234435916884, 0.6168101990561297, 0.6015526229501196, 0.6155480546642632, 0.6047400193356214, 0.5659591214314474, 0.6110860838444383, 0.5850031488967448, 0.5865706335764715, 0.6040032107305141, 0.5811352135774172, 0.6272936560554906, 0.6590505218125586, 0.6590239468564916, 0.6428665059942623, 0.6397459494181232],
        [0.6496810607090193, 0.6024507522586617, 0.5939457503931748, 0.599287735733536, 0.6186263613001032, 0.6559219831932952, 0.643931944920996, 0.6694469192893616, 0.6237737146265019, 0.6221464680242943, 0.5861561705073867, 0.5886360717788865, 0.6119560165261478, 0.5674361604136008, 0.5269785387869066, 0.5292785319973332, 0.5779724964562738, 0.5311427123299441, 0.5063600062606542, 0.5022331146731768, 0.5, 0.5, 0.5, 0.5, 0.5357438439112205, 0.5397141498533881, 0.5128203884445741, 0.5, 0.530136237792602, 0.575952933419885, 0.6116258358751749, 0.5885880527868905, 0.5582147266980118, 0.556199418162077, 0.5395333517901479, 0.5291374424841868, 0.5673681721456123, 0.5272388303762353, 0.5321866656200421, 0.5476050853562985, 0.5694545162072923, 0.6090973454555778, 0.5698610180386162, 0.5246758584542062, 0.5, 0.5412586643977374, 0.5861333470961776, 0.5947227487971033, 0.572358476502725],
        [0.7219076333308321, 0.7294212952872186, 0.7275929162735614, 0.733899475460986, 0.7505431341367086, 0.7814360579783992, 0.812900959320859, 0.7796374575846938, 0.7710402532777632, 0.7603609874031985, 0.7657049150136711, 0.7654282154226613, 0.7674866912264644, 0.796325347880495, 0.7945825195120979, 0.8350359375107594, 0.8131918863615013, 0.8373789425874599, 0.8865558448695404, 0.8458026986862535, 0.7991202204406084, 0.7543395127981434, 0.766567081368534, 0.745290005409746, 0.7446254473153587, 0.7470512168390604, 0.7122705690859139, 0.7322713710972759, 0.7461170628180309, 0.7875098407120591, 0.8045848480585268, 0.822806988142077, 0.8573994514232448, 0.8957500078202011, 0.8459653230098036, 0.8459178760066295, 0.8662005167294178, 0.903513550326957, 0.8602800389517901, 0.8722641019492317, 0.8355785671623486, 0.8750952228449055, 0.8693549894001948, 0.8877335881155558, 0.8882724938500325, 0.8968815127935625, 0.9109379666688314, 0.8790608705804155, 0.8868151223794652],
    ], 60: [
        [0.8145441071190311, 0.8127056074917223, 0.8317251872019263, 0.8657856527868365, 0.8255134810527155, 0.84414686032774, 0.859852619203778, 0.827610412595408, 0.8657616372083176, 0.8800701527875551, 0.9267031766084913, 0.8869021113455617, 0.9296584497553544, 0.9298164661487227, 0.9503836616300039, 0.9932089854802928, 1, 0.9972938550121695, 0.9637435563755579, 0.9597947745797138, 0.9316770515762981, 0.9242050352243417, 0.9452769656843645, 0.9344103858721711, 0.9346046676392163, 0.9174127948720415, 0.9193341498992672, 0.874250404972009, 0.8926611406586797, 0.8572685310379634, 0.8355785155663245, 0.8436180135078089, 0.8905573569057385, 0.8431631316069288, 0.8398313623682535, 0.8227361727401907, 0.8568693918631162, 0.8766935445399698, 0.8571603073837062, 0.9041921697831731, 0.9471624667655175, 0.9065681399182823, 0.9249747767431438, 0.9174914549535456, 0.9444889630322295, 0.917478667744452, 0.9296512468437942, 0.8992594127053958, 0.9117350767822469],
        [0.6831527163560259, 0.6335706215519024, 0.6372601540767318, 0.6, 0.6271155632668293, 0.6007094824133242, 0.6, 0.6, 0.6243598586241081, 0.6034641361680592, 0.6, 0.6, 0.6, 0.6155772914316873, 0.6, 0.6, 0.6112317731990144, 0.6022076931105635, 0.638895463027977, 0.6631925615620977, 0.6189222019083652, 0.6659008005444431, 0.7024406700089507, 0.7517779932402329, 0.7985050202920772, 0.8329751233642384, 0.8511665346592237, 0.8676630658793668, 0.8607319399112826, 0.8348792580989458, 0.8588748144853697, 0.9001030582881613, 0.866673350070917, 0.8822213618120419, 0.8924989658152873, 0.8494623377683708, 0.894581222133482, 0.9148725291791835, 0.9003444797953944, 0.8983418566978759, 0.9266603084563881, 0.8884908477699753, 0.9196265453383006, 0.8835314022065648, 0.8984265612877005, 0.8828384229612255, 0.8500610419126333, 0.8306942450788514, 0.8795023071399265],
        [0.6634213797651269, 0.6844982625255113, 0.639205885732991, 0.6, 0.6, 0.6, 0.6035287135493633, 0.6, 0.6, 0.6, 0.6, 0.6026151603007757, 0.6, 0.6331077201573084, 0.6729745390373155, 0.6522247885409124, 0.7011686537032616, 0.7079754427400112, 0.6990362213907106, 0.7159164355311418, 0.6907571069394004, 0.6959473074562254, 0.6813295658674283, 0.6376310942133889, 0.6681848704878594, 0.6361441133846396, 0.6162092453571317, 0.6632408508756493, 0.6320519156064133, 0.6418726901173845, 0.6414362842620458, 0.6549784541561907, 0.6532299655246762, 0.700940643822475, 0.7170445124952859, 0.672830755432768, 0.6637210990358122, 0.6167072910821279, 0.6180271168622202, 0.6, 0.6368271593096941, 0.6619603333915193, 0.6297933205439666, 0.6371866563045145, 0.6243702651100563, 0.6447988883299542, 0.6019477191373417, 0.624477150402859, 0.6],
        [0.9948910250858436, 0.9677305563891677, 0.9680717076200984, 0.9514889448205005, 0.9049577631536336, 0.8563343502744744, 0.880677724310639, 0.8577381890737142, 0.8967375585795965, 0.9334787291355306, 0.9115684044824139, 0.9089639911285686, 0.8930504215344534, 0.9117673230398344, 0.929608217790094, 0.9778655919467004, 0.9398130116506055, 0.9626734604334533, 0.9721396383200873, 1, 0.9749286529556771, 0.9580087655538184, 1, 1, 1, 1, 1, 1, 1, 0.9664233809980853, 0.9388196842355551, 0.970255513601257, 0.939603397873435, 0.9633004735524644, 0.9662375546948675, 0.9312649438987473, 0.8892119618662495, 0.9222400926192235, 0.9660107567765316, 0.933245200488505, 0.9622636347569091, 0.9253725070752067, 0.9453563620106671, 0.9701100279607311, 0.9227949889198773, 0.8789944272104973, 0.8606289704097683, 0.8718109443874192, 0.9100126673291856]
    ], 70: [
        [0.9480138085474538, 0.9487703440972971, 0.9511067347061134, 0.953749516389911, 0.9511363127209672, 0.9558314968695878, 0.9579140041948319, 0.96229175371205, 0.9642581115114432, 0.9681170008084242, 0.9706105951812615, 0.972338438993139, 0.9765503346164016, 0.9797766374207207, 0.9765845067540098, 0.9767178078019219, 0.9759479303242736, 0.9761451130185179, 0.9740196444461158, 0.9763389528210401, 0.9762261842294097, 0.9790568519018309, 0.9750641736818585, 0.9797110997981686, 0.9839368720356706, 0.9884282577945522, 0.9873293056941926, 0.9865515919447196, 0.9888100341012132, 0.9868112106843453, 0.9872790211223191, 0.9840041790598606, 0.981196145586737, 0.9819225310729408, 0.9836182995992707, 0.9852784957128053, 0.9850255555813688, 0.9813790864630796, 0.9771675219156043, 0.9782471304732181, 0.9795423806574829, 0.9822086138168266, 0.9835962932476602, 0.9792036688080011, 0.9799547769672255, 0.9751522452615876, 0.9790120657939669, 0.9829993110353528, 0.9878223996618544, 0.9829993110353528, 0.9878223996618544],
        [0.9163680114238419, 0.9140592181310503, 0.9184196375912066, 0.9226983350378288, 0.9267520552980681, 0.9291899282178917, 0.9297044117683838, 0.9260508887983835, 0.9274955123253527, 0.9299535514170847, 0.9315770814256896, 0.9344431255328406, 0.9356491616448195, 0.9391532676629601, 0.9414268289819079, 0.9435609290403328, 0.9394095265316253, 0.9377834733155556, 0.9421945642244041, 0.9420327531381621, 0.9443503125386631, 0.9430709944173663, 0.9450273534413881, 0.9410917622490186, 0.9408057196428414, 0.9406808189754869, 0.9445209533877335, 0.9444235477745214, 0.9398164705269142, 0.9417360921960578, 0.9456034965329208, 0.9492468172051868, 0.9531210594296458, 0.9489971121574181, 0.9497656077104123, 0.9466470682932124, 0.9507880142670152, 0.9477783659370388, 0.9451080379792527, 0.9493779403702651, 0.9516755802102899, 0.9534202471849894, 0.9505473071501861, 0.9479317664668686, 0.9490260802511127, 0.9522575413067008, 0.9545246637120612, 0.9540011178918204, 0.9553691080901154, 0.9540011178918204, 0.9553691080901154],
        [0.8606059065686197, 0.8578231380528257, 0.8581755033815562, 0.85341321141294, 0.8571300920286532, 0.8549061345412808, 0.8523529548618232, 0.8494288554050277, 0.8445036889190307, 0.844700528185832, 0.8446399308755548, 0.8461455304774229, 0.8416830767428002, 0.8461218916471519, 0.8418134365206866, 0.8389867419881492, 0.8347752136450596, 0.8343174093726246, 0.8316459788599675, 0.833044626810937, 0.8284532053427721, 0.8291293841591086, 0.8244859728171634, 0.8257103708405966, 0.8236137395745241, 0.8203582301689488, 0.8221647073863637, 0.8248824248455132, 0.8254495997140929, 0.8228372624561916, 0.8234606476415982, 0.8247980427642583, 0.8291203659481216, 0.8267563926127754, 0.8260656346641747, 0.821168853761047, 0.8226167882221362, 0.8242816565991595, 0.8273606601875533, 0.831844849412982, 0.8332373070526146, 0.8310281186672405, 0.8331007726329139, 0.8376369079749341, 0.8365298408962163, 0.8370781997068656, 0.8403923986789522, 0.8439867724793768, 0.8468529874313847, 0.8439867724793768, 0.8468529874313847],
        [0.8344750073040842, 0.8367650460309913, 0.8408291654398146, 0.8417183769010879, 0.8440096868812488, 0.8407962294160061, 0.8454345310032656, 0.8448966725774465, 0.8437754495703471, 0.839736593925105, 0.8432827296755998, 0.8421272048565537, 0.8372925493194293, 0.8334666357455691, 0.8353069869784912, 0.8332410659658647, 0.828738944997586, 0.8301890862281387, 0.8347794576793558, 0.8347454565706908, 0.8333648120656139, 0.8355778510702337, 0.8383949296294652, 0.8354134282617212, 0.8328279854413065, 0.8373106539157185, 0.8389063471750879, 0.8424926307959977, 0.8442870589892921, 0.8432894854891602, 0.8400258351901401, 0.8375313826872683, 0.8408738178696717, 0.8409842374965847, 0.8360560565545873, 0.839256365473696, 0.8360304672389819, 0.8391652419237652, 0.8379669549162246, 0.8372774937831123, 0.8340081342297044, 0.8368812465026891, 0.8359634151102852, 0.8322994990778142, 0.8360367750298351, 0.8353093356539442, 0.835762824031078, 0.8339478856373546, 0.830918041232952, 0.8339478856373546, 0.830918041232952]
    ], 80: [
        [0.9838755394257631, 1, 1, 1, 0.9954805512431495, 1, 1, 1, 0.9829200295992212, 1, 0.9812150965985531, 1, 0.9511237228663675, 0.9043839963902159, 0.8661239308279808, 0.9148012942077659, 0.8862882071804761, 0.8470211915582216, 0.8942112955225892, 0.8870722863616164, 0.8874392898632746, 0.9073295574192185, 0.8968785815655921, 0.856027147807079, 0.8203152517644257, 0.8175814888360589, 0.8, 0.8354398079030544, 0.8092255217385419, 0.8369042380852262, 0.8519436474299182, 0.8386588470310583, 0.8196264850856443, 0.8, 0.8443540995887698, 0.8536800089731915, 0.845098396710241, 0.8151132866098783, 0.8077190510656089, 0.817871871381434, 0.8117206503781893, 0.8238988064282092, 0.8, 0.8359495351753107, 0.8305896930023906, 0.8470503309400821, 0.8594192138680283, 0.8230794421837797, 0.8524538371889104],
        [0.8432126693458223, 0.8469128871708965, 0.8666860486576883, 0.9161259656806439, 0.9433061667669039, 0.9852057106071264, 1, 1, 1, 1, 1, 0.9988655119991041, 1, 1, 0.9753820870836654, 0.9402857165908375, 0.9846756027014119, 0.9791500782054847, 0.9680500281130846, 0.9580819920810265, 0.938807308562077, 0.914400438229744, 0.9342176645358062, 0.9270563208299295, 0.9562518452371562, 1, 0.9706871634140911, 0.9426596863538759, 0.9403263445692057, 0.9696676570705729, 1, 0.9817115405546042, 0.9640147401272542, 0.9899666851651611, 1, 1, 0.9768959352098292, 0.9312823390256716, 0.9728292716792502, 0.9351929948749822, 0.9636476302508299, 0.9976356713186216, 1, 1, 1, 0.9527441778082815, 0.9721289203139569, 0.9568151715931061, 0.9591224334015696],
        [0.9240974320739821, 0.9570038386315443, 0.9617602348116218, 0.9671678646846504, 0.954400532452241, 0.9537166529524175, 0.918192373737023, 0.8978938552488014, 0.9335855142077386, 0.8970886872185455, 0.8773316968085, 0.829128026037533, 0.865520676484421, 0.830850579820906, 0.821141232406912, 0.8, 0.8246678307377437, 0.8507899840056448, 0.8413909575657883, 0.8523956908803539, 0.8974921407714092, 0.8573622638517548, 0.8908196420563768, 0.9224322936558154, 0.8774404538828936, 0.9256759154278003, 0.9192604934643479, 0.9470270460416553, 0.9011905858453234, 0.95028288374685, 0.9305682423619807, 0.9495153479443306, 0.9988341636237456, 0.9980304566030163, 0.9930372695711834, 0.9537453015959131, 0.9406739594940536, 0.9272297917427073, 0.968049777678939, 0.9575273204961756, 0.9286764690540263, 0.89194652672077, 0.9370833408944624, 0.8908138058927174, 0.9118733081149779, 0.8817070495646638, 0.8713777135618661, 0.8757579342660742, 0.9093938598281991],
        [0.8875741478917475, 0.8944765709299711, 0.9232747439736013, 0.923736598361967, 0.9447915219316589, 0.9573811195042576, 0.9118150246114297, 0.8756763754636551, 0.8827151155761398, 0.8445718160099327, 0.8211485458121073, 0.8155042921764235, 0.8, 0.8092453743578023, 0.8, 0.8, 0.8, 0.8010540402193859, 0.8230317862778269, 0.8559607911355763, 0.8710341099265255, 0.8456633824675257, 0.8803848606317954, 0.838174877355016, 0.8730515827966479, 0.9226968064569837, 0.9272900832116602, 0.9169021318736889, 0.8963554401075752, 0.8533122282042552, 0.8118008166866859, 0.8064787315355525, 0.8, 0.8043829042967298, 0.850590569261939, 0.8609106754654746, 0.8455777963570881, 0.815177726983678, 0.8, 0.8130117668211677, 0.8140318362245004, 0.8, 0.8102449936525713, 0.8, 0.8024367424355602, 0.8380595533614814, 0.8, 0.8, 0.8128672020002233]
    ], 90: [
        [0.997815444608025, 0.9709341398985349, 0.9479571442740834, 0.943735856223367, 0.9074444996342579, 0.954716543651977, 0.9658823491849247, 1, 0.9667767215522597, 0.9805183816379995, 0.9324308427684167, 0.9, 0.9367331266959474, 0.9, 0.9, 0.9, 0.9402625796917605, 0.9120611099096334, 0.9590185250984063, 0.9929792691740214, 0.9867247697848957, 0.9956066674134431, 0.964404249414187, 0.9599396497043969, 0.9793220866920959, 0.9912244100110744, 1, 1, 1, 0.9506290733535652, 0.971526073831481, 0.9482763355739424, 0.9121464556554725, 0.9101426847978248, 0.9, 0.9, 0.9, 0.9, 0.9474175515826608, 0.9135800136398784, 0.9185615524319559, 0.9291969375822869, 0.9476745733461754, 0.9210444603612157, 0.9, 0.9, 0.9, 0.9051616713519441, 0.9075569666326544],
        [0.9114743258876828, 0.9009312171307118, 0.9, 0.9, 0.9, 0.9, 0.9485456754406751, 0.9783633632129629, 1, 0.9744147841297044, 0.9636336501249184, 0.9167745293467521, 0.9, 0.9304367449940524, 0.9355855011208447, 0.9468020641542231, 0.9891847036484304, 0.9549973681940797, 0.9117414948369978, 0.9, 0.9484813734396959, 0.9876506983916717, 1, 1, 0.9831509932612886, 0.9512888331355734, 0.9989396801659226, 1, 1, 1, 0.9947799300999233, 0.9837725969682942, 0.9686259475879927, 0.9329372085609456, 0.9102823524283961, 0.9264793367973017, 0.9600926502309227, 0.9493331872858869, 0.9620907192840633, 0.964106412258174, 1, 1, 0.986531063559717, 1, 1, 1, 0.9676130830065767, 1, 0.9541464747564362],
        [0.9280113712977989, 0.9, 0.9257797605854694, 0.960264255722769, 0.9733963270099916, 0.9542007887178188, 0.9765507037352141, 0.9547300435853072, 1, 1, 1, 1, 1, 0.9541104531066358, 0.9467421478583463, 0.9861521993234401, 0.9624822013504113, 0.9600939600709787, 0.9207243794373015, 0.9641664610259231, 0.9844228843056179, 1, 1, 0.9567759885430053, 0.9986138398388087, 0.965053589259092, 0.9989081826383934, 1, 1, 0.9856389070150169, 0.9900068936773416, 1, 1, 1, 1, 0.9823506048333239, 0.9615587950190329, 1, 1, 1, 1, 1, 1, 1, 0.9910518546122926, 0.9759323583301993, 0.996570788931326, 1, 1],
        [0.997215485091831, 0.9478775010713855, 0.9558702853693589, 0.9506650324612573, 0.9582310858904729, 0.9487415389704456, 0.967195576917766, 0.9685565228233309, 0.9655478170192835, 0.9385006750234564, 0.9, 0.9, 0.9, 0.9, 0.9095339644154653, 0.9246969037892476, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9352143233753754, 0.9317620574820993, 0.9180699265192375, 0.9, 0.9, 0.9103200865310935, 0.9275248494878252, 0.9621358942567725, 0.9949400828768798, 0.9733749445212666, 1, 1, 1, 0.9757099907506193, 0.949116429744512, 0.9019916238533642, 0.9417860638733575, 0.981409757209689, 1, 0.969743594587389, 1, 0.995677576343564, 0.9590856048296718, 0.918324461326197, 0.9, 0.9125040806357184, 0.9]
    ]
}


class User:
    """A User getting data from the satellite"""
    def __init__(self, agent_id, snr_min):
        self.agent_id = agent_id

        # self.snr_noise = [np.random.uniform(SNR_NOISE_LOW, 1)]
        self.snr_noise = TMP_SNR[snr_min][self.agent_id]
        self.index = -1

        self.download_log = {}

        self.sat_log = {}

        # just consider downlink for now; more interesting for most apps anyways
        self.log = structlog.get_logger(agent_id=self.agent_id)
        self.log.debug('User init', agent_id=self.agent_id)

    def __repr__(self):
        return str(self.agent_id)

    def get_snr_noise(self, mahimahi_ptr=None):
        # return self.snr_noise[-1]
        if mahimahi_ptr:
            if mahimahi_ptr < 0:
                mahimahi_ptr = 0
            try:
                return self.snr_noise[mahimahi_ptr]
            except IndexError:
                print(len(self.snr_noise))
                print(mahimahi_ptr)
        else:
            return self.snr_noise[self.index]

    def get_snr_log(self):
        return self.snr_noise

    def update_sat_log(self, sat_id, mahimahi_ptr):
        self.sat_log[mahimahi_ptr] = sat_id

    def get_conn_sat_id(self, mahimahi_ptr):
        self.log.debug("get_conn_sat_id", log=self.sat_log, ptr=mahimahi_ptr)
        sat_id = None
        for i in sorted(self.sat_log.keys()):
            if mahimahi_ptr < i:
                break
            sat_id = self.sat_log[i]

        return sat_id

    def update_snr_noise(self):
        self.index += 1
        return self.snr_noise[self.index]

    def get_agent_id(self):
        return self.agent_id

    def update_download(self, mahimahi_ptr, sat_id, video_chunk_remain, quality, last_quality, buf_size):
        self.download_log[mahimahi_ptr] = [sat_id, video_chunk_remain, quality, last_quality, buf_size]

    def get_related_download_logs(self, mahimahi_ptr, target_mahimahi_ptr):
        self.log.debug('download_log', download_log=self.download_log)
        if not self.download_log:
            return [None] * 6
        final_logs = []
        video_chunk_remain = None
        sat_id = None
        last_quality = None
        buf_size = None
        first_mahimahi_ptr = None
        min_idx = None
        ptr_list = sorted(self.download_log.keys())
        for ptr in range(len(ptr_list)):
            if ptr > target_mahimahi_ptr:
                break
            min_idx = ptr
        if min_idx is None or np.abs(mahimahi_ptr - ptr_list[min_idx]) > np.abs(mahimahi_ptr - target_mahimahi_ptr):
            return [None] * 6
        if min_idx != 0:
            last_quality = self.download_log[ptr_list[min_idx-1]][3]

        if video_chunk_remain is None:
            video_chunk_remain = self.download_log[ptr_list[min_idx]][1]
        if sat_id is None:
            sat_id = self.download_log[ptr_list[min_idx]][0]
        if buf_size is None:
            buf_size = self.download_log[ptr_list[min_idx]][4]
        if first_mahimahi_ptr is None:
            first_mahimahi_ptr = ptr_list[min_idx]

        for i in range(min_idx, len(ptr_list)):
            final_logs.append(self.download_log[ptr_list[i]])

        return first_mahimahi_ptr, sat_id, video_chunk_remain, final_logs, last_quality, buf_size
