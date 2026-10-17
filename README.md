# Data Mining Project: จัดกลุ่มและแนะนำเพลงจาก Top Spotify Songs 2023

โครงงานนี้ใช้เทคนิค Data Mining แบบ **Unsupervised Learning (K-means Clustering)** จัดกลุ่มเพลงจากชุดข้อมูล Top Spotify Songs 2023 ใน 2 มุมมอง คือ **ความนิยม** และ **ลักษณะเสียง** แล้วนำกลุ่มที่ได้ไปใช้แนะนำเพลง

## ชุดข้อมูล

- ชื่อ: **Top Spotify Songs 2023** (Kaggle, เผยแพร่โดย Nidula Elgiriyewithana)
- ลิงก์: https://www.kaggle.com/datasets/nelgiriyewithana/top-spotify-songs-2023
- ขนาด: 953 เพลง 24 คอลัมน์ (เหลือ 952 เพลงหลังทำความสะอาด)
- เนื้อหา: ค่าเสียงของเพลง (bpm, danceability, valence, energy, acousticness, instrumentalness, liveness, speechiness) และความนิยมข้ามแพลตฟอร์ม (Spotify, Apple, Deezer, Shazam)
- ไฟล์นี้ต้องอ่านด้วย encoding `latin-1` (เปิดด้วย UTF-8 ไม่ได้)

## ปัญหาที่พบและวิธีจัดการ

| ปัญหา | วิธีจัดการ |
|---|---|
| `streams` เสียหาย 1 แถว | ตัดแถวทิ้ง |
| ตัวเลขบางคอลัมน์เป็นข้อความมีคอมมา เช่น `"1,234"` | ลบคอมมาแล้วแปลงเป็นตัวเลข |
| ข้อมูลความนิยมเบ้มาก | ทำ `log1p` ก่อน Standardize |
| สเกลของฟีเจอร์ต่างกัน (เช่น bpm เทียบกับค่า 0-100%) | Standardize (ค่าเฉลี่ย 0, ส่วนเบี่ยงเบนมาตรฐาน 1) |
| ไม่มี genre และไม่มี label | ใช้ Unsupervised Learning |
| ชื่อเพลง/ศิลปินบางแถวเพี้ยนจากไฟล์ต้นฉบับ (กู้คืนไม่ได้) | ไม่นำมาแสดงในผลแนะนำเพลง |

## วิธีการ

### 1. Popularity Model (จัดกลุ่มตามความนิยม)
- ฟีเจอร์: `streams`, `in_spotify_playlists`, `in_spotify_charts`
- ใช้ playlist/ชาร์ตของ Spotify แพลตฟอร์มเดียว เพราะ playlist ของ Spotify, Apple, Deezer สัมพันธ์กันสูง (0.73-0.85) ถ้าใส่ทั้งหมดจะนับซ้ำ
- `log1p` + Standardize แล้วจัดกลุ่มด้วย K-means (k=3)

### 2. Audio Model (จัดกลุ่มตามลักษณะเสียง)
- ฟีเจอร์ 8 ตัว: `bpm`, `danceability_%`, `valence_%`, `energy_%`, `acousticness_%`, `instrumentalness_%`, `liveness_%`, `speechiness_%`
- Standardize แล้วจัดกลุ่มด้วย K-means (k=3)

### 3. การเลือกค่า k
ทดลอง k = 2-8 แล้วดู Elbow และ Silhouette ที่ k=2 ได้ silhouette สูงกว่าเล็กน้อย แต่แบ่งได้แค่ 2 กลุ่ม จึงเลือก **k=3** เพราะตีความได้ละเอียดกว่า

### 4. การตั้งชื่อกลุ่ม
ตั้งชื่ออัตโนมัติจากโปรไฟล์ของแต่ละกลุ่ม (คลาส `AutoLabel`) ไม่ผูกกับเลขกลุ่ม เพราะเลขกลุ่มของ K-means สลับได้

| โมเดล | ชื่อกลุ่ม | ฟีเจอร์ที่ใช้ตัดสิน |
|---|---|---|
| Popularity | All-Time Megahits | `streams` สูงสุด |
| Popularity | Top Charting Hits | ใน 2 กลุ่มที่เหลือ `in_spotify_charts` สูงกว่า |
| Popularity | Moderate Hits | กลุ่มที่เหลือ |
| Audio | Rap Hip-Hop Talk-Heavy | `speechiness_%` สูงสุด |
| Audio | Acoustic Chill Melancholic | ใน 2 กลุ่มที่เหลือ `acousticness_%` สูงกว่า |
| Audio | Upbeat Dance Party | กลุ่มที่เหลือ |

> ชื่อกลุ่มเป็นการตีความของผู้วิเคราะห์จากตัวเลข ไม่ใช่ผลที่โมเดลให้โดยตรง และกติการองรับเฉพาะ k=3

### 5. การแนะนำเพลง
ฟังก์ชัน `recommend(song, model, count)`

1. หาว่าเพลงเริ่มต้นอยู่ cluster ไหนของโมเดลที่เลือก (`"audio"` หรือ `"pop"`)
2. ดึงเพลงอื่นใน cluster เดียวกันมาเป็นผู้สมัคร
3. เรียงตามระยะห่าง (Euclidean) จากเพลงเริ่มต้น ในปริภูมิฟีเจอร์ที่ Standardize แล้ว
4. คืน `count` เพลงที่ใกล้ที่สุด

| โมเดล | ความหมายของผลแนะนำ |
|---|---|
| `audio` | เพลงที่ **เสียงคล้ายกัน** |
| `pop` | เพลงที่ **ดังระดับเดียวกัน** (ไม่จำเป็นต้องฟังคล้ายกัน) |

พารามิเตอร์เสริม: `max_per_artist` (จำกัดจำนวนเพลงต่อศิลปิน เช่น `1` = ไม่ซ้ำศิลปิน)

## ผลลัพธ์

### Evaluation

| โมเดล | k | Silhouette | หมายเหตุ |
|---|---|---|---|
| Popularity | 3 | 0.367 | แยกกลุ่มชัดกว่า (k=2: 0.376) |
| Audio | 3 | 0.206 | ขอบเขตกลุ่มซ้อนกัน (k=2: 0.228) |

### ขนาดและลักษณะกลุ่ม

| โมเดล | กลุ่ม | จำนวนเพลง | ลักษณะเด่น |
|---|---|---|---|
| Popularity | All-Time Megahits | 329 | streams มัธยฐาน 921 ล้าน, ปีมัธยฐาน 2020 |
| Popularity | Moderate Hits | 382 | streams 190 ล้าน, ไม่ติดชาร์ต |
| Popularity | Top Charting Hits | 241 | ติดชาร์ต 13 ครั้ง, ปีมัธยฐาน 2023 |
| Audio | Upbeat Dance Party | 556 | valence 59.6, energy 72.3 |
| Audio | Acoustic Chill Melancholic | 249 | acousticness 52.4, valence 33.7 |
| Audio | Rap Hip-Hop Talk-Heavy | 147 | speechiness 30.3 |

### ข้อค้นพบ
- เพลงอะคูสติกเศร้าอยู่ในกลุ่ม All-Time Megahits 40% สูงสุด ส่วนแร็พ 23% ต่ำสุด
- เพลงปี 2023 กระจุกอยู่ในกลุ่ม Top Charting Hits
- เป็นแนวโน้มจากการนับสัดส่วน ไม่ได้ทดสอบนัยสำคัญทางสถิติ

### ตัวอย่างการแนะนำเพลง (เพลงเริ่มต้น: Perfect - Ed Sheeran)

| อันดับ | `audio` (เสียงคล้ายกัน) | `pop` (ดังระดับเดียวกัน) |
|---|---|---|
| 1 | All Too Well (Taylor Swift) | Say You Won't Let Go (James Arthur) |
| 2 | Maroon (Taylor Swift) | Shallow (Lady Gaga, Bradley Cooper) |
| 3 | After Hours (The Weeknd) | Circles (Post Malone) |
| 4 | Am I Dreaming (A$AP Rocky, Metro Boomin, Roisee) | Believer (Imagine Dragons) |
| 5 | Creep (Radiohead) | Something Just Like This (The Chainsmokers, Coldplay) |

## วิธีรัน

### สิ่งที่ต้องมี
- Python 3.10 ขึ้นไป
- ไลบรารี: `pandas`, `numpy`, `scikit-learn`, `matplotlib`

```bash
pip install pandas numpy scikit-learn matplotlib
```

### ขั้นตอน
1. ดาวน์โหลด `spotify_2023.csv` จาก Kaggle (ลิงก์ด้านบน)
2. วางไฟล์ไว้โฟลเดอร์เดียวกับ `clustering_all.py` หรือกำหนด path ผ่านตัวแปรแวดล้อม

```bash
python clustering_all.py
```

```bash
# กรณีไฟล์ข้อมูลอยู่ที่อื่น
SPOTIFY_CSV=/path/to/spotify_2023.csv python clustering_all.py      # macOS / Linux
set SPOTIFY_CSV=C:\path\to\spotify_2023.csv && python clustering_all.py   # Windows
```

### ผลลัพธ์ที่โปรแกรมสร้าง

| ไฟล์ | เนื้อหา |
|---|---|
| `chart/evaluation-*.png` | กราฟ Elbow และ Silhouette ของแต่ละโมเดล |
| `chart/cluster-*.png` | กราฟกลุ่ม (PCA 2 มิติ) และจำนวนเพลงต่อกลุ่ม |
| `output/summary.txt` | สรุปผลทั้งหมด (โปรไฟล์กลุ่ม ข้อค้นพบ ตัวอย่างการแนะนำ ข้อจำกัด) |
| `output/spotify_2023_clustered.csv` | ข้อมูลรายเพลงพร้อมชื่อกลุ่มทั้ง 2 แบบ |

### เปลี่ยนเพลงที่ต้องการแนะนำ
แก้ที่ส่วน "แนะนำเพลง" ใกล้ท้ายไฟล์ `clustering_all.py`

```python
print(recommend("Perfect", "audio", 5).to_string(index=False))
print(recommend("Blinding Lights", "pop", 10, max_per_artist=1).to_string(index=False))
```

ชื่อเพลงไม่สนตัวพิมพ์เล็ก/ใหญ่ ถ้าพิมพ์ผิดโปรแกรมจะแสดงชื่อที่ใกล้เคียงให้

## โครงสร้างของโค้ด (`clustering_all.py`)

| คลาส/ฟังก์ชัน | หน้าที่ |
|---|---|
| `DataProcessing` | โหลดและทำความสะอาดข้อมูล |
| `DataModel` | เก็บฟีเจอร์ Standardize หาค่า k (Elbow, Silhouette) |
| `Cluster` | จัดกลุ่มด้วย K-means สรุปโปรไฟล์ ตั้งชื่อกลุ่ม |
| `AutoLabel` | ตั้งชื่อกลุ่มอัตโนมัติจากโปรไฟล์ |
| `Chart` | วาดกราฟ |
| `recommend()` | แนะนำเพลงจากกลุ่มเดียวกับเพลงเริ่มต้น |
| `Summary` | สรุปผลทั้งหมดและบันทึกไฟล์ |

## ข้อจำกัด

- ชุดข้อมูลไม่มี genre และพฤติกรรมผู้ใช้ จึงวัดความแม่นยำของการแนะนำไม่ได้ (ไม่มี ground truth) ตรวจได้ด้วยการฟังเท่านั้น
- Audio silhouette ต่ำ (ประมาณ 0.2) ขอบเขตกลุ่มซ้อนกัน เพลงที่อยู่ใกล้เส้นแบ่งอาจอยู่คนละกลุ่ม
- ผลแนะนำอยู่ภายในกลุ่มเดียวกันเท่านั้น เพลงที่ใกล้มากแต่อยู่อีกฝั่งของเส้นแบ่งจะไม่ถูกแนะนำ
- ข้อมูลมีเฉพาะเพลงที่ฮิตอยู่แล้วในปี 2023 จึงไม่ครอบคลุมเพลงทั่วไป
- ชื่อกลุ่มเป็นการตีความของผู้วิเคราะห์ และ `AutoLabel` รองรับเฉพาะ k=3

## แนวทางต่อยอด
- เพิ่มข้อมูล genre (เช่น จาก Spotify API) เพื่อแนะนำตามแนวเพลง
- รวมผลของ 2 โมเดลในการแนะนำครั้งเดียว
- ใช้ K-Nearest Neighbors ค้นหาเพลงใกล้เคียงทั้งชุดโดยไม่จำกัดอยู่ในกลุ่มเดียว

## ไฟล์นำเสนอ
`spotify_data_mining.pptx` สไลด์นำเสนอโครงงาน
