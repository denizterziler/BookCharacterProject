# Anotasyon Kuralları

## Proje Amacı

Bu proje, Türkçe halk anlatılarında / masallarda karakter ağı çıkarımı yapmak için hafif-altın standart anotasyonlar oluşturmayı amaçlar.

Bu anotasyon tam dilbilimsel çekirdek-gönderim (coreference) anotasyonu değildir. Bu sürümde odaklanılan katmanlar şunlardır:

1. Karakter envanteri
2. Temel alias / adlandırma biçimleri
3. Paragraf düzeyinde karakter varlığı
4. Ana karakter ilişkileri

Bu sürümde tüm zamirler, tüm sıfır özne kullanımları, tüm kelime düzeyi mention anotasyonları ve tüm küçük/ikincil ilişkiler anotlanmaz.

---

# 1. Karakter Tanımı

Bir karakter; anlatıda eyleme katılan, konuşan, zarar veren, zarar gören, yardım eden, tehdit eden, karar veren veya olay örgüsünde işlevi olan insan, hayvan, doğaüstü varlık ya da grup olabilir.

Bir varlık şu durumlarda karakter olarak anotlanabilir:

- eylem yapıyorsa,
- konuşuyorsa,
- saldırıyorsa,
- saldırıya uğruyorsa,
- yardım ediyorsa,
- zarar veriyorsa,
- tehdit ediyorsa,
- karar veriyorsa,
- olay örgüsünde tekrar eden bir rol üstleniyorsa,
- hikâyedeki ana çatışmanın parçasıysa.

Nesneler, mekânlar ve soyut kavramlar karakter olarak anotlanmaz.

Karakter olarak anotlanmayan örnekler:

- dağ
- köy
- köy odası
- yuva
- silah
- tuzak
- ekmek
- et

Bu tür öğeler ortam, nesne veya olay örgüsü unsuru olabilir; ancak kendi başlarına eyleyen karakter değildir.

---

# 2. İnsan Karakterler

Bireysel insan karakterler, hikâyede belirgin bir role sahip olduklarında anotlanır.

İnsan grupları da kolektif şekilde hareket ediyorsa grup karakter olarak anotlanabilir.

Örnek:

- “köylüler” grup karakter olarak anotlanır, çünkü birlikte hareket ederler, kurdu yakalarlar, sorgularlar, beslerler ve ondan korkarlar.

---

# 3. Hayvan Karakterler

Hayvanlar, anlatı eyleminde önemli bir rol oynuyorsa karakter olarak anotlanır.

Hayvanlar şu durumlarda karakter olarak alınır:

- konuşuyorsa,
- saldırıyorsa,
- saldırıya uğruyorsa,
- geri dönüyorsa,
- yardım ediyorsa,
- zarar veriyorsa,
- olay örgüsünde bireysel olarak önemli hale geliyorsa.

Genel arka plan hayvanları, bireysel olarak önemli hale gelmedikçe ayrı karakter olarak anotlanmaz.

Örnekler:

- “Eşek” ayrı karakter olarak anotlanır, çünkü kurt tarafından saldırıya uğrar ve eve geri döner.
- “Sıpa” ayrı karakter olarak anotlanır, çünkü eşekle birlikte saldırıya uğrar.
- “Köy hayvanları” grup karakter olarak anotlanır, çünkü kurtun köye verdiği genel zararın merkezinde bu hayvanlar vardır.

---

# 4. Grup Karakterler

Bir grup, hikâyede kolektif bir katılımcı gibi davranıyorsa karakter olarak anotlanabilir.

Örnek grup karakterler:

- Köylüler
- Köy hayvanları

Grup karakterler dikkatli kullanılmalıdır. Eğer grubun içinden bir üye daha sonra bireysel olarak önemli hale gelirse, o üye ayrı karakter olarak anotlanır.

Kural:

Bir grup üyesi daha sonra olay örgüsünde bireysel olarak önemli hale gelirse, ayrı karakter olarak anotlanır ve aynı yüzey biçimi hem grup hem de bireysel karakter için alias olarak kullanılmaz.

Örnek:

- “Köy hayvanları” grup karakterdir.
- “Eşek” ayrı bireysel hayvan karakterdir.
- Bu yüzden “eşek”, “Köy hayvanları” için alias olarak kullanılmaz.

---

# 5. Alias Kuralları

Alias, aynı karaktere gönderim yapan sabit yüzey biçimidir.

Alias olarak şunlar kullanılabilir:

- özel adlar,
- rol adları,
- betimleyici adlandırmalar,
- tekrar eden isim tamlamaları,
- belirli karakteri açıkça gösteren kalıplaşmış ifadeler.

Zamirler bu hafif-altın anotasyon sürümünde alias olarak eklenmez.

Örnek aliaslar:

Kurt için:

- kurt
- canavar
- yaratık
- kurnaz kurt
- ziyankâr yaratık

Köylüler için:

- köylüler
- odadakiler
- fakir fukara

Zamir veya zamirimsi ifadeler alias olarak eklenmez:

- o
- onlar
- sizi
- bizler
- onu
- ona

Bu tür ifadeler gerekirse daha sonra ayrı bir coreference/mention anotasyonunda ele alınabilir.

---

# 6. Paragraf Düzeyinde Karakter Varlığı

Paragraf düzeyinde karakter varlığı, bir karakterin ilgili paragrafta açıkça geçtiği, eyleme katıldığı veya bağlamdan açıkça anlaşıldığı anlamına gelir.

Bir karakter şu durumlarda paragrafta var kabul edilebilir:

- karakterin adı geçiyorsa,
- alias kullanılıyorsa,
- karakter bir eylemin açık öznesiyse,
- karakter diyalogda konuşuyorsa,
- karaktere doğrudan hitap ediliyorsa,
- karakter zamirle veya bağlamla açıkça belirtiliyorsa,
- önceki paragraftan gelen açık özne devamlılığı varsa.

Paragraf varlığı, kelime düzeyi mention anotasyonu gerektirmez.

Örnek:

Bir paragrafta kurt “sizi de yerim” diyorsa, “sizi” köylülere gönderim yapar. Bu durumda “Köylüler” o paragrafta var kabul edilebilir.

Ancak sadece mekân adları karakter varlığı doğurmaz.

Örnek:

Metinde sadece “köy odası” geçiyorsa, bu otomatik olarak “Köylüler” karakterinin paragrafta var olduğu anlamına gelmez.

---

# 7. İlişki Anotasyonu

Sadece ana ilişkiler anotlanır.

Bir ilişkinin anotlanması için metinsel kanıt bulunmalıdır.

Zayıf, belirsiz veya yorum gerektiren ilişkiler atlanabilir.

Her ilişki şu alanları içerir:

- source
- target
- relation_type
- valence
- strength
- evidence_paragraphs
- evidence_text

---

# 8. İlişki Türleri

Bu sürümde kullanılabilecek ilişki türleri:

- enemy/conflict
- helper
- family
- romantic/marriage
- authority
- servant/master
- betrayal
- reward/benefactor
- ownership/caretaker
- acquaintance
- same_group
- unknown

Gerekirse ileride yeni ilişki türleri eklenebilir. Ancak eklenen her yeni tür bu dosyada açıklanmalıdır.

---

# 9. Valence Etiketleri

Kullanılabilecek valence etiketleri:

- positive
- negative
- neutral
- mixed
- unknown

Tanımlar:

- positive: destekleyici, koruyucu, yardım edici veya olumlu ilişki
- negative: zarar verici, tehdit edici, düşmanca veya çatışmalı ilişki
- neutral: işlevsel veya rol temelli, belirgin olumlu/olumsuz yönü olmayan ilişki
- mixed: hem olumlu hem olumsuz yönleri bulunan ilişki
- unknown: yeterli kanıt bulunmayan ilişki

---

# 10. Strength Etiketleri

Kullanılabilecek strength etiketleri:

- weak
- moderate
- strong

Tanımlar:

- weak: ilişki kısa veya zayıf biçimde geçiyorsa
- moderate: ilişki açıkça destekleniyor ama hikâyenin ana ekseni değilse
- strong: ilişki hikâyenin merkezindeyse veya birden fazla paragrafta güçlü biçimde destekleniyorsa

---

# 11. Evidence Text

Evidence text Türkçe yazılmalıdır.

Evidence text, ilişkinin neden anotlandığını kısa ve açık biçimde açıklamalıdır.

Tüm paragrafı alıntılamak zorunda değildir. Ancak metinsel dayanağı anlaşılır şekilde özetlemelidir.

Örnek:

“Kurt köyün hayvanlarını yer; köylüler onu yakalayıp köy odasına getirir ve sorgular. Daha sonra kurt, et getirmezlerse köylüleri de yiyeceğini söyleyerek onları tehdit eder.”

---

# 12. T01 Özel Kararları: Kurt ve Köylüler

T01 için aşağıdaki karakter kararları verilmiştir.

## Dahil edilen karakterler

### C01 — Kurt

Gerekçe: Ana hayvan karakterdir. Eylem yapar, konuşur, tehdit eder, pazarlık yapar ve sonunda kaçar.

### C02 — Köylüler

Gerekçe: Kolektif insan grubudur. Kurdu takip eder, yakalar, sorgular, besler ve ondan korkar.

### C03 — Köy hayvanları

Gerekçe: Kolektif hayvan grubudur. Kurtun köye verdiği genel zarar bu grup üzerinden anlatılır.

### C04 — Eşek

Gerekçe: Bireysel olarak önemli hayvandır. Kurt tarafından saldırıya uğrar ve eve geri döner.

### C05 — Sıpa

Gerekçe: Bireysel olarak önemli hayvandır. Eşekle birlikte kurt tarafından saldırıya uğrar.

### C06 — Eşeğin sahibi

Gerekçe: Eşek ve sıpa ile ilişkili insan karakterdir. Hayvanların zarar gördüğünü anlayınca köy odasına gider.

---

# 13. T01 İçin Dahil Edilmeyen Öğeler

Aşağıdaki öğeler karakter olarak anotlanmamıştır:

- dağ
- köy
- köy odası
- yuva
- silah
- tuzak
- ekmek
- et

Gerekçe: Bunlar mekân veya nesnedir. Hikâyede eyleyen karakter gibi davranmazlar.

---

# 14. T01 İçin Özel Grup Kararı

“Köy hayvanları” genel grup karakteridir.

Bu karaktere dahil edilen aliaslar:

- hayvanlar
- köyün hayvanları
- keçi
- at
- dana

Ancak “eşek” ve “sıpa” bu gruba alias olarak eklenmemiştir.

Gerekçe:

Eşek ve sıpa daha sonra bireysel olarak önemli hale gelir. Bu nedenle ayrı karakterler olarak anotlanırlar.

Bu karar, aynı yüzey biçiminin iki farklı karaktere bağlanmasını önlemek için verilmiştir.

---

# 15. T01 İçin Paragraf Kararları

## P003

P003 içinde “köylüler” kelimesi doğrudan geçmez. Ancak “dağı taramaya koyulmuşlar”, “kurdu izlemişler”, “kurdu görmüşler”, “tuzak kurmuşlar” gibi fiiller P002’deki köylülere gönderim yapar.

Bu nedenle C02 — Köylüler, P003 içinde var kabul edilmiştir.

## P009

P009 içinde “köy odası” geçer, fakat bu sadece mekândır. “Köylüler” açıkça geçmez.

Bu nedenle C02 — Köylüler, P009 içinde var kabul edilmemiştir.

P009 içinde şu karakterler vardır:

- C01 — Kurt
- C04 — Eşek
- C05 — Sıpa
- C06 — Eşeğin sahibi

## P012

P012 içinde kurt “sizi de yerim” der. “Sizi” köylülere gönderim yapar.

Bu nedenle C02 — Köylüler, P012 içinde var kabul edilmiştir.

## P014

P014 içinde kurt ve köy hayvanları açıkça referanslanır. “Onlar ermiş muradına” ifadesi köylülere yorumlanabilir; ancak bu ifade formüle yakın ve belirsiz olduğu için C02 — Köylüler, P014 içinde zorunlu olarak anotlanmamıştır.

---

# 16. T01 İlişki Kararları

T01 içinde şu ana ilişkiler anotlanmıştır:

## C01 — C02

Kurt ile Köylüler arasında güçlü bir enemy/conflict ilişkisi vardır.

Gerekçe: Kurt köyün hayvanlarını yer; köylüler onu yakalar ve sorgular; kurt daha sonra köylüleri de yemekle tehdit eder.

## C01 — C03

Kurt ile Köy hayvanları arasında güçlü bir enemy/conflict ilişkisi vardır.

Gerekçe: Kurt köydeki hayvanları yer ve bu durum hikâyedeki temel çatışmayı oluşturur.

## C01 — C04

Kurt ile Eşek arasında orta güçte enemy/conflict ilişkisi vardır.

Gerekçe: Kurt eşeğe saldırır ve ondan et yer.

## C01 — C05

Kurt ile Sıpa arasında orta güçte enemy/conflict ilişkisi vardır.

Gerekçe: Kurt sıpaya saldırır ve ondan et yer.

## C06 — C04

Eşeğin sahibi ile Eşek arasında ownership/caretaker ilişkisi vardır.

Gerekçe: Eşek eve döndüğünde sahibinin yüreği dağlanır; bu, sahibinin eşekle bağlı olduğunu gösterir.

## C06 — C05

Eşeğin sahibi ile Sıpa arasında ownership/caretaker ilişkisi vardır.

Gerekçe: Sıpa, eşekle birlikte eve döner ve olay sahibini üzer.

## C02 — C03

Köylüler ile Köy hayvanları arasında ownership/caretaker ilişkisi vardır.

Gerekçe: Köy hayvanları köylülerin geçim düzeninin parçasıdır. Köylüler, hayvanlarının yenmesinden dolayı kurda karşı harekete geçer.

---

# 17. Anotasyon Durumu

T01, manuel olarak kontrol edilmiş ve hafif-altın anotasyon olarak kabul edilmiştir.

Bu anotasyon şunları içerir:

- karakter envanteri
- paragraf düzeyinde karakter varlığı
- ana ilişkiler

Bu anotasyon şunları içermez:

- tam kelime düzeyi mention anotasyonu
- tam zamir çözümlemesi
- tam sıfır özne çözümlemesi
- tüm küçük/ikincil ilişkiler