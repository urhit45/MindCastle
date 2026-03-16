export type Lang = 'en' | 'hi'

export interface BilingualText {
  en: string
  hi: string
}

export interface Item {
  label: BilingualText
  stat: BilingualText
  statScore: number
  source: string
}

export interface Category {
  tier: BilingualText
  items: Item[]
}

export const categories: Category[] = [
  {
    tier: { en: '🧱 Bare Necessities', hi: '🧱 बुनियादी ज़रूरतें' },
    items: [
      {
        label: { en: 'Clean Drinking Water Access', hi: 'स्वच्छ पेयजल की उपलब्धता' },
        stat: {
          en: '~600M Indians lack safe drinking water (NITI Aayog, 2023)',
          hi: 'लगभग 60 करोड़ भारतीयों को सुरक्षित पेयजल नहीं मिलता (नीति आयोग, 2023)',
        },
        statScore: 4,
        source: 'NITI Aayog',
      },
      {
        label: { en: 'Sanitation & Open Defecation', hi: 'स्वच्छता और खुले में शौच' },
        stat: {
          en: '~50% rural households lack functional toilets despite Swachh Bharat (WHO/UNICEF 2023)',
          hi: 'स्वच्छ भारत के बावजूद ~50% ग्रामीण घरों में कार्यात्मक शौचालय नहीं (WHO/UNICEF 2023)',
        },
        statScore: 4,
        source: 'WHO/UNICEF',
      },
      {
        label: { en: 'Hunger & Nutrition', hi: 'भूख और पोषण' },
        stat: {
          en: "India ranks 111/125 on Global Hunger Index 2023 — 'serious' level",
          hi: "भारत वैश्विक भूख सूचकांक 2023 में 125 में से 111वें स्थान पर — 'गंभीर' स्तर",
        },
        statScore: 3,
        source: 'Global Hunger Index 2023',
      },
      {
        label: { en: 'Child Malnutrition', hi: 'बच्चों में कुपोषण' },
        stat: {
          en: '35.5% children under 5 are stunted (NFHS-5, 2021)',
          hi: '5 साल से कम उम्र के 35.5% बच्चे अविकसित हैं (NFHS-5, 2021)',
        },
        statScore: 3,
        source: 'NFHS-5',
      },
      {
        label: { en: 'Electricity Access', hi: 'बिजली की उपलब्धता' },
        stat: {
          en: '99.9% village electrification achieved, but supply reliability varies (MoP 2023)',
          hi: '99.9% गाँवों में विद्युतीकरण हुआ, लेकिन आपूर्ति की विश्वसनीयता अलग-अलग है (MoP 2023)',
        },
        statScore: 7,
        source: 'Ministry of Power',
      },
    ],
  },
  {
    tier: { en: '🏫 Rights & Institutions', hi: '🏫 अधिकार और संस्थाएं' },
    items: [
      {
        label: { en: 'Press Freedom', hi: 'प्रेस की स्वतंत्रता' },
        stat: {
          en: 'India ranks 161/180 on Press Freedom Index 2023 (RSF)',
          hi: 'भारत प्रेस स्वतंत्रता सूचकांक 2023 में 180 में से 161वें स्थान पर (RSF)',
        },
        statScore: 2,
        source: 'Reporters Without Borders',
      },
      {
        label: { en: 'Judicial Independence', hi: 'न्यायिक स्वतंत्रता' },
        stat: {
          en: 'India ranks 89/142 on Rule of Law Index 2023 (World Justice Project)',
          hi: 'भारत कानून के शासन सूचकांक 2023 में 142 में से 89वें स्थान पर (WJP)',
        },
        statScore: 5,
        source: 'World Justice Project',
      },
      {
        label: { en: 'Free & Fair Elections', hi: 'स्वतंत्र और निष्पक्ष चुनाव' },
        stat: {
          en: "India rated 'Electoral Autocracy' since 2018 by V-Dem Institute 2023",
          hi: "V-Dem संस्थान ने 2018 से भारत को 'चुनावी तानाशाही' की श्रेणी में रखा है (2023)",
        },
        statScore: 4,
        source: 'V-Dem Institute',
      },
      {
        label: { en: 'Corruption in Government', hi: 'सरकारी भ्रष्टाचार' },
        stat: {
          en: 'India scores 39/100 on Corruption Perceptions Index 2023 (Transparency International)',
          hi: 'भारत का भ्रष्टाचार बोध सूचकांक 2023 में स्कोर 100 में से 39 (Transparency International)',
        },
        statScore: 4,
        source: 'Transparency International',
      },
      {
        label: { en: 'Civil Liberties', hi: 'नागरिक स्वतंत्रताएं' },
        stat: {
          en: "India rated 'Partly Free' with score 66/100 by Freedom House 2023",
          hi: "Freedom House ने भारत को 100 में 66 अंक के साथ 'आंशिक रूप से स्वतंत्र' रेट किया (2023)",
        },
        statScore: 5,
        source: 'Freedom House',
      },
    ],
  },
  {
    tier: { en: '📚 Education & Health', hi: '📚 शिक्षा और स्वास्थ्य' },
    items: [
      {
        label: { en: 'Public Education Quality', hi: 'सार्वजनिक शिक्षा की गुणवत्ता' },
        stat: {
          en: 'India ranks 132/191 on Human Development Index (UNDP 2023)',
          hi: 'भारत मानव विकास सूचकांक में 191 में से 132वें स्थान पर (UNDP 2023)',
        },
        statScore: 5,
        source: 'UNDP HDI 2023',
      },
      {
        label: { en: 'Healthcare Access', hi: 'स्वास्थ्य सेवा की उपलब्धता' },
        stat: {
          en: 'India spends only 2.1% of GDP on public health vs. WHO recommended 5% (MoHFW 2023)',
          hi: 'भारत सार्वजनिक स्वास्थ्य पर GDP का केवल 2.1% खर्च करता है, WHO की सिफारिश 5% है (2023)',
        },
        statScore: 4,
        source: 'Ministry of Health',
      },
      {
        label: { en: "Women's Safety", hi: 'महिला सुरक्षा' },
        stat: {
          en: 'India ranks 127/146 on Gender Gap Index 2023 (WEF)',
          hi: 'भारत लैंगिक अंतर सूचकांक 2023 में 146 में से 127वें स्थान पर (WEF)',
        },
        statScore: 3,
        source: 'World Economic Forum',
      },
      {
        label: { en: 'Infant Mortality', hi: 'शिशु मृत्यु दर' },
        stat: {
          en: "28 deaths per 1,000 live births — improved but still above global average (World Bank 2023)",
          hi: 'प्रति 1,000 जीवित जन्मों पर 28 मौतें — सुधरा है लेकिन वैश्विक औसत से अधिक (World Bank 2023)',
        },
        statScore: 5,
        source: 'World Bank',
      },
    ],
  },
  {
    tier: { en: '💰 Economy & Opportunity', hi: '💰 अर्थव्यवस्था और अवसर' },
    items: [
      {
        label: { en: 'Income Inequality', hi: 'आय असमानता' },
        stat: {
          en: 'Top 1% owns 40.1% of national wealth — highest in recorded history (World Inequality Lab 2024)',
          hi: 'शीर्ष 1% के पास राष्ट्रीय संपत्ति का 40.1% — दर्ज इतिहास में सबसे अधिक (World Inequality Lab 2024)',
        },
        statScore: 2,
        source: 'World Inequality Lab',
      },
      {
        label: { en: 'Youth Unemployment', hi: 'युवा बेरोजगारी' },
        stat: {
          en: 'Youth unemployment at ~45.4% (CMIE 2023)',
          hi: 'युवा बेरोजगारी दर लगभग 45.4% (CMIE 2023)',
        },
        statScore: 3,
        source: 'CMIE',
      },
      {
        label: { en: 'GDP Growth & Development', hi: 'GDP वृद्धि और विकास' },
        stat: {
          en: "6.3% GDP growth — one of world's fastest growing major economies (IMF 2023)",
          hi: '6.3% GDP वृद्धि — दुनिया की सबसे तेज़ बढ़ती प्रमुख अर्थव्यवस्थाओं में से एक (IMF 2023)',
        },
        statScore: 8,
        source: 'IMF',
      },
      {
        label: { en: 'Tax Money Accountability', hi: 'कर धन की जवाबदेही' },
        stat: {
          en: 'CAG audit compliance below 60%; electoral bonds declared unconstitutional (Supreme Court 2024)',
          hi: 'CAG ऑडिट अनुपालन 60% से कम; चुनावी बॉन्ड को सुप्रीम कोर्ट ने असंवैधानिक घोषित किया (2024)',
        },
        statScore: 3,
        source: 'CAG / Supreme Court',
      },
    ],
  },
  {
    tier: { en: '🌐 Global Standing & Future', hi: '🌐 वैश्विक स्थिति और भविष्य' },
    items: [
      {
        label: { en: 'Climate & Environment', hi: 'जलवायु और पर्यावरण' },
        stat: {
          en: 'India ranks 176/180 on Environmental Performance Index 2022 (Yale)',
          hi: 'भारत पर्यावरण प्रदर्शन सूचकांक 2022 में 180 में से 176वें स्थान पर (Yale)',
        },
        statScore: 2,
        source: 'Yale EPI',
      },
      {
        label: { en: 'Digital Infrastructure (UPI)', hi: 'डिजिटल बुनियादी ढांचा (UPI)' },
        stat: {
          en: 'UPI processes 10B+ transactions/month; India leads globally in real-time digital payments',
          hi: 'UPI प्रति माह 10 अरब+ लेनदेन करता है; भारत रियल-टाइम डिजिटल भुगतान में विश्व नेता है',
        },
        statScore: 9,
        source: 'NPCI 2023',
      },
      {
        label: { en: 'Space & Science (ISRO)', hi: 'अंतरिक्ष और विज्ञान (ISRO)' },
        stat: {
          en: "Chandrayaan-3 landed on Moon's south pole — 4th country ever (ISRO 2023)",
          hi: 'चंद्रयान-3 चंद्रमा के दक्षिणी ध्रुव पर उतरा — ऐसा करने वाला चौथा देश (ISRO 2023)',
        },
        statScore: 9,
        source: 'ISRO',
      },
      {
        label: { en: 'Internet Freedom & Shutdowns', hi: 'इंटरनेट स्वतंत्रता और बंद' },
        stat: {
          en: 'India leads world in internet shutdowns — 84 in 2023 alone (Access Now)',
          hi: 'भारत इंटरनेट बंद में दुनिया में अव्वल — 2023 में अकेले 84 बार (Access Now)',
        },
        statScore: 2,
        source: 'Access Now',
      },
    ],
  },
]

export const t = {
  en: {
    appTitle: 'India Report Card',
    appSubtitle: 'भारत रिपोर्ट कार्ड',
    langPrompt: 'Choose your language / अपनी भाषा चुनें',
    welcomeTitle: 'India Democracy Report Card',
    welcomeSubtitle: 'How well do you know India\'s reality?',
    instructions: [
      'Rate each indicator from 1–10 based on your perception.',
      'Then reveal the actual data and see how you compare.',
      'At the end, get your full report card.',
    ],
    startBtn: 'Start →',
    revealBtn: 'Reveal Data',
    yourRating: 'Your Rating',
    dataScore: 'Data Score',
    source: 'Source',
    yourAvg: 'Your Avg',
    dataAvg: 'Data Avg',
    gap: 'Gap',
    verdict: 'Verdict',
    category: 'Category',
    resultsTitle: 'Your India Report Card',
    resultsSubtitle: 'How your perception compares to reality',
    overconfident: '🔴 Overconfident',
    underestimated: '🟢 Underestimated',
    aligned: '🟡 Aligned',
    shareBtn: 'Share Results',
    startOver: 'Start Over',
    copiedMsg: 'Copied to clipboard!',
    overestimated: 'Overestimated',
    underestimatedItem: 'Underestimated',
    onPoint: 'On Point',
    viewResults: 'View Results →',
    allRevealed: 'Reveal all items to see results',
    revealAll: 'Reveal All & See Results',
    langToggle: 'हि',
    ratingLabel: (v: number) => `${v}/10`,
  },
  hi: {
    appTitle: 'भारत रिपोर्ट कार्ड',
    appSubtitle: 'India Report Card',
    langPrompt: 'अपनी भाषा चुनें / Choose your language',
    welcomeTitle: 'भारत लोकतंत्र रिपोर्ट कार्ड',
    welcomeSubtitle: 'क्या आप भारत की हकीकत जानते हैं?',
    instructions: [
      'प्रत्येक संकेतक को 1–10 के पैमाने पर अपनी राय के अनुसार रेट करें।',
      'फिर असली डेटा देखें और जानें कि आप कहाँ खड़े हैं।',
      'अंत में अपना पूरा रिपोर्ट कार्ड प्राप्त करें।',
    ],
    startBtn: 'शुरू करें →',
    revealBtn: 'डेटा देखें',
    yourRating: 'आपकी रेटिंग',
    dataScore: 'डेटा स्कोर',
    source: 'स्रोत',
    yourAvg: 'आपका औसत',
    dataAvg: 'डेटा औसत',
    gap: 'अंतर',
    verdict: 'निर्णय',
    category: 'श्रेणी',
    resultsTitle: 'आपका भारत रिपोर्ट कार्ड',
    resultsSubtitle: 'आपकी धारणा बनाम हकीकत',
    overconfident: '🔴 अति-आत्मविश्वास',
    underestimated: '🟢 कम आंका',
    aligned: '🟡 संतुलित',
    shareBtn: 'परिणाम शेयर करें',
    startOver: 'फिर से शुरू करें',
    copiedMsg: 'क्लिपबोर्ड पर कॉपी हो गया!',
    overestimated: 'ज़्यादा आंका',
    underestimatedItem: 'कम आंका',
    onPoint: 'सटीक',
    viewResults: 'परिणाम देखें →',
    allRevealed: 'परिणाम देखने के लिए सभी आइटम खोलें',
    revealAll: 'सब खोलें और परिणाम देखें',
    langToggle: 'EN',
    ratingLabel: (v: number) => `${v}/10`,
  },
}
