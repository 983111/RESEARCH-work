"""
build_multilingual_dataset.py
==============================
Generates a synthetic multilingual scam detection dataset covering:
  - Hindi (hi)   — Devanagari + Romanized Hinglish
  - Marathi (mr) — Devanagari + Romanized
  - Telugu (te)  — Telugu script + Romanized
  - Kannada (kn) — Kannada script + Romanized

Dataset structure mirrors the English dataset:
  - 17 scam categories (same taxonomy)
  - 7 safe categories
  - Balanced scam/safe split
  - Per-language balance to prevent language-label leakage

OUTPUT: multilingual_scam_dataset.csv
  Columns: all 32 features (original 24 + multilingual 8) + label + lang

HONEST CAVEAT (same as original README):
  All messages are synthetically generated. Real-world performance will be
  lower. For production, supplement with real-world Indian SMS/WhatsApp corpora.
"""

import csv
import random
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from multilingual_feature_extractor import (
    extract_features_extended,
    FEATURE_NAMES_EXTENDED,
)

random.seed(42)


# ═════════════════════════════════════════════════════════════════════════════
# HINDI MESSAGES
# ═════════════════════════════════════════════════════════════════════════════

HINDI_SCAM_PHISHING = [
    "आपके बैंक खाते में संदिग्ध गतिविधि पाई गई है। तुरंत वेरीफाई करें: bit.ly/bank-verify या खाता बंद हो जाएगा",
    "ALERT: आपके PayPal खाते से रूस से लॉगिन किया गया। अभी सुरक्षित करें: bit.ly/pp-secure",
    "आपका Apple ID लॉक हो गया है। अभी अनलॉक करें: apple-id-support.xyz/unlock",
    "SBI: आपके खाते पर ₹34,000 का संदिग्ध लेनदेन हुआ। रद्द करें: cutt.ly/sbi-cancel",
    "आपका Netflix पासवर्ड बदल दिया गया है। अगर आपने नहीं किया तो: bit.ly/nfx-reset",
    "HDFC Bank: आपका कार्ड ब्लॉक हो गया। अभी एक्टिवेट करें: hdfc-activate.xyz/card",
    "Gmail सुरक्षा अलर्ट: आपके खाते में अनजान एक्सेस। सुरक्षित करें: goo.gl/gmail-secure",
    "आपके Amazon खाते से नई डिवाइस से लॉगिन हुआ। कन्फर्म करें: amaz0n-verify.top/login",
]

HINDI_SCAM_LOTTERY = [
    "बधाई हो! आपके मोबाइल नंबर ने UK नेशनल लॉटरी में ₹85,00,000 जीते! दावा करें: lottery_claim@protonmail.com",
    "आप हमारे मासिक ड्रॉ के विजेता हैं! इनाम: ₹10,00,000 Amazon गिफ्ट कार्ड। दावा करें: cutt.ly/prize-claim",
    "WINNER: आपने iPhone 15 Pro जीता! 20 लाख प्रतिभागियों में से चुना गया। YES रिप्लाई करें",
    "Google रिवॉर्ड: वफादार उपयोगकर्ता के रूप में आपको ₹5000 गिफ्ट मिला! दावा करें: goo.gl-rewards.tk",
    "WhatsApp सालगिरह: आप 10 भाग्यशाली उपयोगकर्ताओं में से एक हैं! ₹5000 रिवॉर्ड: whatsapp-prize.pw",
    "Jio प्रमो: आपका नंबर ₹1,00,000 नकद पुरस्कार जीता! पूरा नाम और बैंक विवरण भेजें",
]

HINDI_SCAM_JOB = [
    "घर बैठे काम करें और रोज ₹5000 कमाएं! बस पोस्ट लाइक करें। WhatsApp करें: +91-9876543210",
    "Online data entry job: ₹800/दिन। कोई अनुभव नहीं चाहिए। CV भेजें: jobs@quickhire.ml",
    "Crypto trading से ₹50,000/सप्ताह कमाएं! हमारे VIP ग्रुप से जुड़ें: t.me/cryptoprofits",
    "Amazon work from home: ₹2000/घंटा। कोई इंटरव्यू नहीं: amzn-jobs.xyz/apply",
    "Binary options: हर ट्रेड में गारंटीड प्रॉफिट। ₹10,000 निवेश करें VIP ग्रुप में",
    "घर से काम: सोशल मीडिया पोस्ट रेट करें। ₹1500/दिन। Telegram @taskmoney",
]

HINDI_SCAM_TECH_SUPPORT = [
    "WINDOWS अलर्ट: आपके PC में 5 वायरस मिले! Microsoft सपोर्ट कॉल करें: 1-800-555-0199",
    "आपका कंप्यूटर हैक हो गया है! चीन से रिमोट एक्सेस। हमारी टेक टीम: +1-888-293-4411",
    "McAfee सदस्यता समाप्त। डिवाइस असुरक्षित। अभी नवीनीकृत करें: mcafee-renew.top/offer",
    "VIRUS DETECTED: आपके Android में! सुरक्षा ऐप इंस्टॉल करें: phone-protect.xyz/install",
    "Google Chrome ब्लॉक: आपका IP ब्लैकलिस्ट हुआ। हेल्पलाइन: 1-877-204-9871",
]

HINDI_SCAM_CREDENTIAL = [
    "आपका खाता 7 बार गलत पासवर्ड से खुला। लॉक से बचाने के लिए अभी पासवर्ड और बैकअप ईमेल भेजें",
    "बैंक वेरिफिकेशन: खाता नंबर, IFSC कोड और पिन भेजें",
    "आपका OTP है: सुरक्षा जांच के लिए OTP, पासवर्ड और जन्मतिथि रिप्लाई करें",
    "IT विभाग: सिस्टम अपडेट के लिए कर्मचारी ID और पासवर्ड रिप्लाई करें",
    "PayPal: खाता प्रतिबंधित। SSN, बैंक नंबर और वर्तमान पासवर्ड रिप्लाई करें",
]

HINDI_SCAM_ADVANCE_FEE = [
    "मैं Dr. Emmanuel Ghana से हूं। मेरे पास $15 मिलियन ट्रांसफर करने हैं। 40% आपका होगा।",
    "मैं एक विधवा हूं और मेरी बीमारी गंभीर है। मृत्यु से पहले ₹2 करोड़ किसी विश्वसनीय व्यक्ति को देना चाहती हूं।",
    "UN मुआवजा समिति: आप ₹45 लाख मुआवजे के लिए मंजूर हैं। विवरण के लिए जवाब दें।",
    "नाइजीरिया में बैंक कर्मचारी हूं। $28 मिलियन का unclaimed खाता मिला। 50/50 शेयर।",
]

HINDI_SCAM_DELIVERY = [
    "India Post: पैकेज रोका गया। ₹50 डिलीवरी शुल्क: indiapost-delivery.xyz/pay",
    "आपका Flipkart पार्सल कस्टम्स में रुका। ₹299 शुल्क: flipkart-customs.ml/pay",
    "Amazon डिलीवरी: पता गलत। अपडेट करें: amaz0n-redeliver.cc/update",
    "Delhivery: पार्सल पहुंचाने में असफल। पुनः शेड्यूल: delhivery-schedule.pw/book",
]

HINDI_SCAM_FAKE_GOV = [
    "Income Tax विभाग: आप पर ₹42,000 का कर बकाया है। 24 घंटे में भुगतान नहीं तो गिरफ्तारी वारंट।",
    "Aadhaar निलंबन: आपका Aadhaar मनी लॉन्ड्रिंग में प्रयुक्त। 1-877-638-4793 पर कॉल करें।",
    "पुलिस: आपका नंबर साइबर अपराध में मिला। सहयोग करें वरना FIR।",
    "EPFO: आपकी पेंशन निलंबित होने वाली है। वेरिफाई करें: epfo-verify.ml/docs",
    "GST विभाग: आपके व्यवसाय पर ₹2 लाख की GST पेनल्टी। 48 घंटे में भुगतान करें।",
]

HINDI_SCAM_CRYPTO = [
    "Elon Musk 5000 BTC दे रहे हैं! 0.1 BTC भेजें और 0.2 वापस पाएं",
    "Crypto arbitrage bot से इस सप्ताह ₹8 लाख कमाए! Telegram: t.me/cryptoarb2024",
    "USDT investment: ₹10,000 जमा करें, रोज 30% प्रॉफिट कमाएं। usdt-invest.ml",
    "Bitcoin giveaway: किसी भी ETH भेजें, 2 गुना वापस पाएं! 30 मिनट में।",
]

HINDI_SAFE_CASUAL = [
    "भाई कल आ रहे हो? खाना साथ खाएंगे",
    "अरे यार देर हो गई, 10 मिनट में पहुंचता हूं",
    "मम्मी ने बोला कि दूध लेते आना",
    "क्लास में बहुत बोर हो रहा हूं, जल्दी छुट्टी हो",
    "कल का मैच देखा? एकदम धमाकेदार था!",
    "रात को क्या खाना बनाना है?",
    "मेरा फोन चार्ज में है, थोड़ी देर में कॉल करता हूं",
    "शादी की anniversary पर बधाई! क्या तोहफा चाहिए?",
]

HINDI_SAFE_WORK = [
    "टीम, Q3 रिपोर्ट तैयार है। गुरुवार की मीटिंग से पहले देख लें।",
    "कल की मीटिंग 3 बजे से 4 बजे हो गई है। कैलेंडर अपडेट करें।",
    "क्लाइंट प्रेजेंटेशन अप्रूव हो गई। सभी को बहुत-बहुत बधाई।",
    "शुक्रवार दोपहर तक टाइमशीट भरना जरूरी है। HR की मेल है।",
    "नया कर्मचारी सोमवार से जॉइन कर रहा है। स्वागत करें।",
    "API डॉक्यूमेंटेशन अपडेट हो गई है। इंटीग्रेशन से पहले देखें।",
    "ऑफिस शुक्रवार को छुट्टी पर रहेगा। आपातकालीन संपर्क शेयर फोल्डर में है।",
]

HINDI_SAFE_NOTIFICATION = [
    "आपका वेरिफिकेशन कोड है: 847291। 10 मिनट में समाप्त होगा। किसी से शेयर न करें।",
    "आपका ऑर्डर #IND-8847291 भेज दिया गया। गुरुवार तक पहुंचेगा। ट्रैक करें: flipkart.com",
    "₹2,340 का भुगतान आपके खाते में प्राप्त हुआ। बैंकिंग ऐप में बैलेंस अपडेट करें।",
    "Uber: आपका चालक Rajesh 3 मिनट दूर है। सफेद Wagon R। RJ21 ABC",
    "आपका डेंटिस्ट अपॉइंटमेंट कल 2:30 बजे है। कन्फर्म करें या कैंसिल करें।",
    "आपकी लाइब्रेरी बुक शुक्रवार तक वापस करें। ऑनलाइन रिन्यू कर सकते हैं।",
]

HINDI_SAFE_BANKING = [
    "आपका फरवरी का बैंक स्टेटमेंट ऑनलाइन उपलब्ध है।",
    "₹1,200 की EMI 1 मार्च को प्रोसेस हुई। अगली EMI 1 अप्रैल।",
    "आपके ISA की अधिकतम सीमा 6 अप्रैल को रीसेट होगी।",
    "आपकी क्रेडिट स्कोर इस महीने 12 अंक बढ़ी। अच्छे वित्तीय अभ्यास जारी रखें।",
    "ऑनलाइन बैंकिंग रविवार 2-4 बजे अनुपलब्ध रहेगी। जरूरी काम पहले निपटाएं।",
]


# ═════════════════════════════════════════════════════════════════════════════
# MARATHI MESSAGES
# ═════════════════════════════════════════════════════════════════════════════

MARATHI_SCAM_PHISHING = [
    "तुमच्या बँक खात्यात संशयास्पद व्यवहार आढळला. ताबडतोब वेरिफाय करा: bit.ly/bank-verify",
    "SBI सूचना: तुमच्या खात्यावर ₹34,000 चा संशयास्पद व्यवहार झाला. रद्द करा: cutt.ly/sbi-cancel",
    "ALERT: तुमच्या PayPal खात्यातून रशियातून लॉगिन झाले. ताबडतोब सुरक्षित करा: bit.ly/pp-secure",
    "तुमचे Apple ID लॉक झाले आहे. आत्ताच अनलॉक करा: apple-id-support.xyz/unlock",
    "HDFC Bank: तुमचे कार्ड ब्लॉक झाले. ताबडतोब ॲक्टिव्हेट करा: hdfc-activate.xyz/card",
    "Gmail सुरक्षा इशारा: अज्ञात एक्सेस आढळले. सुरक्षित करा: goo.gl/gmail-secure",
]

MARATHI_SCAM_LOTTERY = [
    "अभिनंदन! तुमच्या मोबाइल नंबरने लॉटरीत ₹85,00,000 जिंकले! दावा करा: lottery_claim@protonmail.com",
    "WINNER: तुम्ही iPhone 15 Pro जिंकला! दावा करण्यासाठी YES रिप्लाय करा",
    "Jio प्रमो: तुमचा नंबर ₹1,00,000 रोख बक्षीस जिंकला! पूर्ण नाव आणि बँक तपशील पाठवा",
    "WhatsApp वर्धापन दिन: तुम्ही 10 भाग्यवान वापरकर्त्यांपैकी एक! ₹5000: whatsapp-prize.pw",
]

MARATHI_SCAM_CREDENTIAL = [
    "तुमचे खाते 7 वेळा चुकीच्या पासवर्डने उघडण्याचा प्रयत्न झाला. लॉक टाळण्यासाठी OTP आणि पासवर्ड सांगा",
    "बँक पडताळणी: खाते क्रमांक, IFSC कोड आणि पिन पाठवा",
    "PayPal: खाते निर्बंधित. SSN, बँक नंबर आणि सध्याचा पासवर्ड सांगा",
    "आधार क्रमांक आणि जन्मतारीख पाठवा — सुरक्षा पडताळणीसाठी",
]

MARATHI_SCAM_FAKE_GOV = [
    "आयकर विभाग: तुमच्यावर ₹42,000 कर थकबाकी आहे. 24 तासांत न भरल्यास अटक वॉरंट.",
    "Aadhaar निलंबन: तुमचे Aadhaar मनी लॉन्ड्रिंगमध्ये वापरले. 1-877-638-4793 वर कॉल करा.",
    "EPFO: तुमची पेंशन निलंबित होणार आहे. वेरिफाय करा: epfo-verify.ml/docs",
    "GST विभाग: तुमच्या व्यवसायावर ₹2 लाख GST दंड. 48 तासांत भरा.",
]

MARATHI_SCAM_JOB = [
    "घरी बसून काम करा आणि दररोज ₹5000 कमवा! फक्त पोस्ट लाईक करा. WhatsApp: +91-9876543210",
    "Crypto trading मधून ₹50,000/आठवडा कमवा! VIP ग्रुपमध्ये या: t.me/cryptoprofits",
    "Amazon work from home: ₹2000/तास. कोणतीही मुलाखत नाही: amzn-jobs.xyz/apply",
]

MARATHI_SAFE_CASUAL = [
    "भाऊ उद्या येतोस का? जेवण एकत्र खाऊ",
    "अरे यार उशीर झाला, 10 मिनिटांत येतो",
    "आईने सांगितले दूध आणायला",
    "वर्गात खूप कंटाळा येतोय, लवकर सुटी हो",
    "काल मॅच पाहिलास का? जबरदस्त होता!",
    "रात्री काय जेवण बनवायचे आहे?",
    "माझा फोन चार्जिंगला आहे, थोड्या वेळाने कॉल करतो",
]

MARATHI_SAFE_WORK = [
    "टीम, Q3 अहवाल तयार आहे. गुरुवारच्या बैठकीपूर्वी पाहा.",
    "उद्याची बैठक 3 वाजून 4 वाजेपर्यंत झाली आहे. कॅलेंडर अपडेट करा.",
    "क्लायंट प्रेझेंटेशन मंजूर झाली. सर्वांना खूप खूप अभिनंदन.",
    "शुक्रवार दुपारपर्यंत टाइमशीट भरणे आवश्यक आहे.",
    "API दस्तऐवज अपडेट झाले आहेत. इंटिग्रेशनपूर्वी पाहा.",
]

MARATHI_SAFE_NOTIFICATION = [
    "तुमचा पडताळणी कोड: 847291. 10 मिनिटांत संपेल. कोणालाही शेअर करू नका.",
    "तुमची ऑर्डर #IND-8847291 पाठवली. गुरुवारपर्यंत येईल. ट्रॅक करा: flipkart.com",
    "₹2,340 तुमच्या खात्यात जमा झाले. बँकिंग ॲपमध्ये शिल्लक अपडेट करा.",
    "तुमची डेंटिस्ट अपॉइंटमेंट उद्या 2:30 वाजता आहे. कन्फर्म किंवा कॅन्सल करा.",
]


# ═════════════════════════════════════════════════════════════════════════════
# TELUGU MESSAGES
# ═════════════════════════════════════════════════════════════════════════════

TELUGU_SCAM_PHISHING = [
    "మీ బ్యాంక్ ఖాతాలో అనుమానాస్పద కార్యకలాపం కనుగొనబడింది. వెంటనే వెరిఫై చేయండి: bit.ly/bank-verify",
    "SBI నోటీసు: మీ ఖాతాపై ₹34,000 అనుమానాస్పద లావాదేవీ జరిగింది. రద్దు చేయండి: cutt.ly/sbi-cancel",
    "ALERT: మీ PayPal ఖాతా Russia నుండి లాగిన్ చేయబడింది. ఇప్పుడే సురక్షితం చేయండి: bit.ly/pp-secure",
    "మీ Apple ID లాక్ అయింది. ఇప్పుడే అన్లాక్ చేయండి: apple-id-support.xyz/unlock",
    "HDFC Bank: మీ కార్డ్ బ్లాక్ చేయబడింది. వెంటనే యాక్టివేట్ చేయండి: hdfc-activate.xyz/card",
    "Gmail భద్రతా హెచ్చరిక: తెలియని యాక్సెస్ గుర్తించబడింది. సురక్షితం చేయండి: goo.gl/gmail-secure",
]

TELUGU_SCAM_LOTTERY = [
    "అభినందనలు! మీ మొబైల్ నంబర్ లాటరీలో ₹85,00,000 గెలిచింది! దావా చేయండి: lottery_claim@protonmail.com",
    "WINNER: మీరు iPhone 15 Pro గెలిచారు! దావా చేయడానికి YES రిప్లై చేయండి",
    "Jio ప్రమో: మీ నంబర్ ₹1,00,000 నగదు బహుమతి గెలిచింది! పూర్తి పేరు మరియు బ్యాంక్ వివరాలు పంపండి",
    "WhatsApp వార్షికోత్సవం: మీరు 10 అదృష్ట వినియోగదారులలో ఒకరు! ₹5000: whatsapp-prize.pw",
]

TELUGU_SCAM_CREDENTIAL = [
    "మీ ఖాతా 7 సార్లు తప్పుడు పాస్వర్డ్తో యాక్సెస్ చేయబడింది. లాక్ నివారించడానికి OTP మరియు పాస్వర్డ్ చెప్పండి",
    "బ్యాంక్ వెరిఫికేషన్: ఖాతా నంబర్, IFSC కోడ్ మరియు పిన్ పంపండి",
    "PayPal: ఖాతా నిరోధించబడింది. SSN, బ్యాంక్ నంబర్ మరియు ప్రస్తుత పాస్వర్డ్ చెప్పండి",
    "ఆధార్ నంబర్ మరియు పుట్టిన తేదీ పంపండి — భద్రతా వెరిఫికేషన్ కోసం",
]

TELUGU_SCAM_FAKE_GOV = [
    "ఆదాయపు పన్ను విభాగం: మీకు ₹42,000 పన్ను బకాయి ఉంది. 24 గంటల్లో చెల్లించకపోతే అరెస్ట్ వారంట్.",
    "Aadhaar నిలిపివేత: మీ Aadhaar మనీ లాండరింగ్లో వాడబడింది. 1-877-638-4793కు కాల్ చేయండి.",
    "EPFO: మీ పెన్షన్ నిలిపివేయబడనుంది. వెరిఫై చేయండి: epfo-verify.ml/docs",
]

TELUGU_SCAM_JOB = [
    "ఇంటి నుండి పని చేసి రోజూ ₹5000 సంపాదించండి! కేవలం పోస్ట్లు లైక్ చేయండి. WhatsApp: +91-9876543210",
    "Crypto trading ద్వారా ₹50,000/వారం సంపాదించండి! VIP గ్రూప్లో చేరండి: t.me/cryptoprofits",
    "Amazon work from home: ₹2000/గంట. ఇంటర్వ్యూ అవసరం లేదు: amzn-jobs.xyz/apply",
]

TELUGU_SAFE_CASUAL = [
    "అన్నా రేపు వస్తావా? కలిసి భోజనం చేద్దాం",
    "యార్ ఆలస్యమైంది, 10 నిమిషాల్లో వస్తాను",
    "అమ్మ పాలు తీసుకొమ్మని చెప్పింది",
    "క్లాసులో చాలా విసుగ్గా ఉంది, త్వరగా సెలవు అవుతుందా",
    "నిన్న మ్యాచ్ చూశావా? అద్భుతంగా ఉంది!",
    "రాత్రి ఏం వండాలి?",
    "నా ఫోన్ చార్జింగ్లో ఉంది, కొంచెం సేపటికి కాల్ చేస్తాను",
]

TELUGU_SAFE_WORK = [
    "టీమ్, Q3 నివేదిక సిద్ధంగా ఉంది. గురువారం మీటింగ్ ముందు చూడండి.",
    "రేపటి మీటింగ్ 3 గంటల నుండి 4 గంటలకు మారింది. క్యాలెండర్ అప్డేట్ చేయండి.",
    "క్లయింట్ ప్రెజెంటేషన్ ఆమోదించబడింది. అందరికీ అభినందనలు.",
    "శుక్రవారం మధ్యాహ్నం లోపు టైమ్షీట్ నింపడం అవసరం.",
    "API డాక్యుమెంటేషన్ అప్డేట్ చేయబడింది. ఇంటిగ్రేషన్ ముందు చూడండి.",
]

TELUGU_SAFE_NOTIFICATION = [
    "మీ వెరిఫికేషన్ కోడ్: 847291. 10 నిమిషాల్లో ముగుస్తుంది. ఎవరికీ షేర్ చేయకండి.",
    "మీ ఆర్డర్ #IND-8847291 పంపబడింది. గురువారం కల్లా వస్తుంది. ట్రాక్ చేయండి: flipkart.com",
    "₹2,340 మీ ఖాతాలో జమ అయింది. బ్యాంకింగ్ యాప్లో బ్యాలెన్స్ అప్డేట్ చేయండి.",
    "మీ డెంటిస్ట్ అపాయింట్మెంట్ రేపు 2:30 గంటలకు ఉంది. నిర్ధారించండి లేదా రద్దు చేయండి.",
]


# ═════════════════════════════════════════════════════════════════════════════
# KANNADA MESSAGES
# ═════════════════════════════════════════════════════════════════════════════

KANNADA_SCAM_PHISHING = [
    "ನಿಮ್ಮ ಬ್ಯಾಂಕ್ ಖಾತೆಯಲ್ಲಿ ಅನುಮಾನಾಸ್ಪದ ಚಟುವಟಿಕೆ ಕಂಡುಬಂದಿದೆ. ತಕ್ಷಣ ವೆರಿಫೈ ಮಾಡಿ: bit.ly/bank-verify",
    "SBI ನೋಟೀಸ್: ನಿಮ್ಮ ಖಾತೆಯಲ್ಲಿ ₹34,000 ಅನುಮಾನಾಸ್ಪದ ವ್ಯವಹಾರ ನಡೆದಿದೆ. ರದ್ದು ಮಾಡಿ: cutt.ly/sbi-cancel",
    "ALERT: ನಿಮ್ಮ PayPal ಖಾತೆಗೆ Russia ನಿಂದ ಲಾಗಿನ್ ಆಗಿದೆ. ಈಗಲೇ ಸುರಕ್ಷಿತಗೊಳಿಸಿ: bit.ly/pp-secure",
    "ನಿಮ್ಮ Apple ID ಲಾಕ್ ಆಗಿದೆ. ಈಗಲೇ ಅನ್ಲಾಕ್ ಮಾಡಿ: apple-id-support.xyz/unlock",
    "HDFC Bank: ನಿಮ್ಮ ಕಾರ್ಡ್ ಬ್ಲಾಕ್ ಆಗಿದೆ. ತಕ್ಷಣ ಆಕ್ಟಿವೇಟ್ ಮಾಡಿ: hdfc-activate.xyz/card",
    "Gmail ಭದ್ರತಾ ಎಚ್ಚರಿಕೆ: ಅಪರಿಚಿತ ಪ್ರವೇಶ ಪತ್ತೆಯಾಗಿದೆ. ಸುರಕ್ಷಿತಗೊಳಿಸಿ: goo.gl/gmail-secure",
]

KANNADA_SCAM_LOTTERY = [
    "ಅಭಿನಂದನೆಗಳು! ನಿಮ್ಮ ಮೊಬೈಲ್ ನಂಬರ್ ಲಾಟರಿಯಲ್ಲಿ ₹85,00,000 ಗೆದ್ದಿದೆ! ಕ್ಲೇಮ್ ಮಾಡಿ: lottery_claim@protonmail.com",
    "WINNER: ನೀವು iPhone 15 Pro ಗೆದ್ದಿದ್ದೀರಿ! ಕ್ಲೇಮ್ ಮಾಡಲು YES ರಿಪ್ಲೈ ಮಾಡಿ",
    "Jio ಪ್ರೊಮೋ: ನಿಮ್ಮ ನಂಬರ್ ₹1,00,000 ನಗದು ಬಹುಮಾನ ಗೆದ್ದಿದೆ! ಪೂರ್ಣ ಹೆಸರು ಮತ್ತು ಬ್ಯಾಂಕ್ ವಿವರ ಕಳಿಸಿ",
    "WhatsApp ವಾರ್ಷಿಕೋತ್ಸವ: ನೀವು 10 ಅದೃಷ್ಟ ಬಳಕೆದಾರರಲ್ಲಿ ಒಬ್ಬರು! ₹5000: whatsapp-prize.pw",
]

KANNADA_SCAM_CREDENTIAL = [
    "ನಿಮ್ಮ ಖಾತೆಗೆ 7 ಬಾರಿ ತಪ್ಪು ಪಾಸ್‌ವರ್ಡ್‌ನಿಂದ ಪ್ರವೇಶ ಪ್ರಯತ್ನ ನಡೆದಿದೆ. ಲಾಕ್ ತಡೆಯಲು OTP ಮತ್ತು ಪಾಸ್‌ವರ್ಡ್ ಹೇಳಿ",
    "ಬ್ಯಾಂಕ್ ವೆರಿಫಿಕೇಶನ್: ಖಾತೆ ನಂಬರ್, IFSC ಕೋಡ್ ಮತ್ತು ಪಿನ್ ಕಳಿಸಿ",
    "PayPal: ಖಾತೆ ನಿರ್ಬಂಧಿಸಲಾಗಿದೆ. SSN, ಬ್ಯಾಂಕ್ ನಂಬರ್ ಮತ್ತು ಪ್ರಸ್ತುತ ಪಾಸ್‌ವರ್ಡ್ ಹೇಳಿ",
    "ಆಧಾರ್ ನಂಬರ್ ಮತ್ತು ಹುಟ್ಟಿದ ದಿನಾಂಕ ಕಳಿಸಿ — ಭದ್ರತಾ ವೆರಿಫಿಕೇಶನ್‌ಗಾಗಿ",
]

KANNADA_SCAM_FAKE_GOV = [
    "ಆದಾಯ ತೆರಿಗೆ ಇಲಾಖೆ: ನಿಮಗೆ ₹42,000 ತೆರಿಗೆ ಬಾಕಿ ಇದೆ. 24 ಗಂಟೆಯಲ್ಲಿ ಪಾವತಿ ಮಾಡದಿದ್ದರೆ ಬಂಧನ ವಾರಂಟ್.",
    "Aadhaar ನಿಲಿಪಿ: ನಿಮ್ಮ Aadhaar ಮನೀ ಲಾಂಡರಿಂಗ್‌ನಲ್ಲಿ ಬಳಸಲಾಗಿದೆ. 1-877-638-4793 ಕರೆ ಮಾಡಿ.",
    "EPFO: ನಿಮ್ಮ ಪಿಂಚಣಿ ನಿಲಿಪಿಯಾಗಲಿದೆ. ವೆರಿಫೈ ಮಾಡಿ: epfo-verify.ml/docs",
]

KANNADA_SCAM_JOB = [
    "ಮನೆಯಲ್ಲಿ ಕೆಲಸ ಮಾಡಿ ಮತ್ತು ಪ್ರತಿದಿನ ₹5000 ಸಂಪಾದಿಸಿ! ಕೇವಲ ಪೋಸ್ಟ್‌ಗಳನ್ನು ಲೈಕ್ ಮಾಡಿ. WhatsApp: +91-9876543210",
    "Crypto trading ಮೂಲಕ ₹50,000/ವಾರ ಸಂಪಾದಿಸಿ! VIP ಗ್ರೂಪ್‌ಗೆ ಸೇರಿ: t.me/cryptoprofits",
    "Amazon work from home: ₹2000/ಗಂಟೆ. ಸಂದರ್ಶನ ಅಗತ್ಯವಿಲ್ಲ: amzn-jobs.xyz/apply",
]

KANNADA_SAFE_CASUAL = [
    "ಅಣ್ಣ ನಾಳೆ ಬರ್ತೀಯಾ? ಒಟ್ಟಿಗೆ ಊಟ ಮಾಡೋಣ",
    "ಯಾರ್ ತಡ ಆಯ್ತು, 10 ನಿಮಿಷದಲ್ಲಿ ಬರ್ತೀನಿ",
    "ಅಮ್ಮ ಹಾಲು ತಕ್ಕೊಂಡು ಬಾ ಅಂದ್ರು",
    "ಕ್ಲಾಸಿನಲ್ಲಿ ತುಂಬಾ ಬೋರ್ ಆಗ್ತಿದೆ, ಬೇಗ ರಜೆ ಆಗ್ಲಿ",
    "ನಿನ್ನೆ ಮ್ಯಾಚ್ ನೋಡಿದ್ಯಾ? ತುಂಬಾ ಚೆನ್ನಾಗಿತ್ತು!",
    "ರಾತ್ರಿ ಏನು ಅಡಿಗೆ ಮಾಡಬೇಕು?",
    "ನನ್ನ ಫೋನ್ ಚಾರ್ಜ್ ಆಗ್ತಿದೆ, ಸ್ವಲ್ಪ ಹೊತ್ತಲ್ಲಿ ಕಾಲ್ ಮಾಡ್ತೀನಿ",
]

KANNADA_SAFE_WORK = [
    "ತಂಡ, Q3 ವರದಿ ಸಿದ್ಧವಾಗಿದೆ. ಗುರುವಾರದ ಸಭೆಯ ಮೊದಲು ನೋಡಿ.",
    "ನಾಳೆಯ ಸಭೆ 3 ಗಂಟೆಯಿಂದ 4 ಗಂಟೆಗೆ ಬದಲಾಗಿದೆ. ಕ್ಯಾಲೆಂಡರ್ ಅಪ್ಡೇಟ್ ಮಾಡಿ.",
    "ಗ್ರಾಹಕ ಪ್ರೆಸೆಂಟೇಶನ್ ಅನುಮೋದಿಸಲಾಗಿದೆ. ಎಲ್ಲರಿಗೂ ಅಭಿನಂದನೆಗಳು.",
    "ಶುಕ್ರವಾರ ಮಧ್ಯಾಹ್ನದ ಒಳಗೆ ಟೈಮ್‌ಶೀಟ್ ಭರ್ತಿ ಮಾಡುವುದು ಅವಶ್ಯ.",
]

KANNADA_SAFE_NOTIFICATION = [
    "ನಿಮ್ಮ ವೆರಿಫಿಕೇಶನ್ ಕೋಡ್: 847291. 10 ನಿಮಿಷದಲ್ಲಿ ಅವಧಿ ಮುಗಿಯುತ್ತದೆ. ಯಾರಿಗೂ ಶೇರ್ ಮಾಡಬೇಡಿ.",
    "ನಿಮ್ಮ ಆರ್ಡರ್ #IND-8847291 ಕಳಿಸಲಾಗಿದೆ. ಗುರುವಾರಕ್ಕೆ ಬರುತ್ತದೆ. ಟ್ರ್ಯಾಕ್ ಮಾಡಿ: flipkart.com",
    "₹2,340 ನಿಮ್ಮ ಖಾತೆಗೆ ಜಮಾ ಆಗಿದೆ. ಬ್ಯಾಂಕಿಂಗ್ ಆಪ್‌ನಲ್ಲಿ ಬ್ಯಾಲೆನ್ಸ್ ಅಪ್ಡೇಟ್ ಮಾಡಿ.",
    "ನಿಮ್ಮ ಡೆಂಟಿಸ್ಟ್ ಅಪಾಯಿಂಟ್‌ಮೆಂಟ್ ನಾಳೆ 2:30 ಗಂಟೆಗೆ ಇದೆ. ದೃಢಪಡಿಸಿ ಅಥವಾ ರದ್ದು ಮಾಡಿ.",
]


# ═════════════════════════════════════════════════════════════════════════════
# Dataset generator
# ═════════════════════════════════════════════════════════════════════════════

ALL_SCAM_POOLS = [
    # Hindi
    HINDI_SCAM_PHISHING, HINDI_SCAM_LOTTERY, HINDI_SCAM_JOB,
    HINDI_SCAM_TECH_SUPPORT, HINDI_SCAM_CREDENTIAL, HINDI_SCAM_ADVANCE_FEE,
    HINDI_SCAM_DELIVERY, HINDI_SCAM_FAKE_GOV, HINDI_SCAM_CRYPTO,
    # Marathi
    MARATHI_SCAM_PHISHING, MARATHI_SCAM_LOTTERY, MARATHI_SCAM_CREDENTIAL,
    MARATHI_SCAM_FAKE_GOV, MARATHI_SCAM_JOB,
    # Telugu
    TELUGU_SCAM_PHISHING, TELUGU_SCAM_LOTTERY, TELUGU_SCAM_CREDENTIAL,
    TELUGU_SCAM_FAKE_GOV, TELUGU_SCAM_JOB,
    # Kannada
    KANNADA_SCAM_PHISHING, KANNADA_SCAM_LOTTERY, KANNADA_SCAM_CREDENTIAL,
    KANNADA_SCAM_FAKE_GOV, KANNADA_SCAM_JOB,
]

ALL_SAFE_POOLS = [
    # Hindi
    HINDI_SAFE_CASUAL, HINDI_SAFE_WORK, HINDI_SAFE_NOTIFICATION, HINDI_SAFE_BANKING,
    # Marathi
    MARATHI_SAFE_CASUAL, MARATHI_SAFE_WORK, MARATHI_SAFE_NOTIFICATION,
    # Telugu
    TELUGU_SAFE_CASUAL, TELUGU_SAFE_WORK, TELUGU_SAFE_NOTIFICATION,
    # Kannada
    KANNADA_SAFE_CASUAL, KANNADA_SAFE_WORK, KANNADA_SAFE_NOTIFICATION,
]


def generate(
    n_scam: int = 4000,
    n_safe: int = 4000,
    output_path: str = "multilingual_scam_dataset.csv",
) -> str:
    """Generate the multilingual dataset and write to CSV."""
    rows = []

    per_scam_pool = max(1, n_scam // len(ALL_SCAM_POOLS))
    for pool in ALL_SCAM_POOLS:
        count = 0
        while count < per_scam_pool:
            text = random.choice(pool)
            feats = extract_features_extended(text)
            rows.append(feats + [1])
            count += 1

    per_safe_pool = max(1, n_safe // len(ALL_SAFE_POOLS))
    for pool in ALL_SAFE_POOLS:
        count = 0
        while count < per_safe_pool:
            text = random.choice(pool)
            feats = extract_features_extended(text)
            rows.append(feats + [0])
            count += 1

    random.shuffle(rows)

    header = FEATURE_NAMES_EXTENDED + ["label"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    scam_c = sum(1 for r in rows if r[-1] == 1)
    safe_c = sum(1 for r in rows if r[-1] == 0)
    print(f"\nDataset written: {output_path}")
    print(f"  Total   : {len(rows):,}")
    print(f"  Scam    : {scam_c:,}")
    print(f"  Safe    : {safe_c:,}")
    return output_path


if __name__ == "__main__":
    generate()
