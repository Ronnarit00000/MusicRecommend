"""
โครงงาน Data Mining: จัดกลุ่มเพลงและแนะนำเพลงจากชุดข้อมูล Top Spotify Songs 2023
ชุดข้อมูล: https://www.kaggle.com/datasets/nelgiriyewithana/top-spotify-songs-2023

แนวคิด
------
ชุดข้อมูลไม่มี label และไม่มี genre จึงใช้ Unsupervised Learning (K-means Clustering)
จัดกลุ่มเพลง 2 มุมมอง แล้วนำกลุ่มที่ได้ไปแนะนำเพลง

    1) Popularity Model : จัดกลุ่มตามความนิยม (streams, จำนวน playlist, จำนวนครั้งที่ติดชาร์ต)
    2) Audio Model      : จัดกลุ่มตามลักษณะเสียง (bpm, danceability, valence, energy ฯลฯ)

ขั้นตอนการทำงาน
---------------
    1. Data Preprocessing : โหลดข้อมูล ทำความสะอาด
    2. Feature Engineering: เลือกฟีเจอร์ ทำ log1p (เฉพาะความนิยม) และ Standardize
    3. Model              : หาค่า k (Elbow + Silhouette) แล้วจัดกลุ่มด้วย K-means
    4. Interpretation     : สรุปโปรไฟล์กลุ่ม แล้วตั้งชื่อกลุ่มอัตโนมัติจากโปรไฟล์
    5. Recommendation     : recommend(song, model, count) แนะนำเพลงจากกลุ่มเดียวกัน
    6. Summary            : สรุปผลทั้งหมดเป็นข้อความ บันทึกเป็นไฟล์

วิธีรัน
-------
    pip install pandas numpy scikit-learn matplotlib
    python main.py
    (วาง spotify_2023.csv ไว้โฟลเดอร์เดียวกัน หรือกำหนด path ผ่านตัวแปรแวดล้อม SPOTIFY_CSV)

ผลลัพธ์
-------
    chart/                      กราฟ Elbow/Silhouette และกราฟกลุ่ม (PCA) ของทั้ง 2 โมเดล
    output/summary.txt          สรุปผลทั้งหมด
    output/spotify_2023_clustered.csv   ข้อมูลรายเพลงพร้อมชื่อกลุ่มทั้ง 2 แบบ
"""
import os
import time
import difflib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# --------------------------------------------------------------------------
# ค่าคงที่ที่ใช้หลายที่
# --------------------------------------------------------------------------
RANDOM_STATE = 42   # ตรึงค่าสุ่ม ให้รันซ้ำแล้วได้ผลเหมือนเดิม

# ชื่อกลุ่มความนิยม (ใช้ทั้งตอนตั้งชื่อและตอนสรุปผล)
LABEL_MEGA = "All-Time Megahits"       # ดังที่สุด สะสมยอดมานาน
LABEL_CHARTING = "Top Charting Hits"   # ติดชาร์ตสูง (เป็นเพลงใหม่เป็นส่วนใหญ่)
LABEL_MODERATE = "Moderate Hits"       # ดังระดับกลาง ไม่ค่อยติดชาร์ต

# ข้อความที่ปรากฏเมื่อไฟล์ต้นฉบับเสียหายตอนอ่านด้วย latin-1 (กู้ชื่อกลับมาไม่ได้)
GARBLED_MARK = "ï¿½"


# ==========================================================================
# 1) DataProcessing: โหลดและทำความสะอาดข้อมูล
# ==========================================================================
class DataProcessing:
   @staticmethod
   def load_data(path: str) -> pd.DataFrame:
      # ไฟล์นี้เปิดด้วย UTF-8 ไม่ได้ (ชื่อเพลงมีอักขระพิเศษ) จึงต้องใช้ latin-1
      return pd.read_csv(path, encoding='latin-1')

   @staticmethod
   def clean_data(data_frame: pd.DataFrame) -> pd.DataFrame:
      # คอลัมน์ที่ควรเป็นตัวเลข แต่บางคอลัมน์ถูกเก็บเป็นข้อความ เช่น "1,234"
      numeric_cols = [
         'streams',
         'in_deezer_playlists',
         'in_shazam_charts',
         'in_spotify_playlists',
         'in_spotify_charts',
         'in_apple_playlists',
         'in_apple_charts',
         'in_deezer_charts',
         'bpm',
         'danceability_%',
         'valence_%',
         'energy_%',
         'acousticness_%',
         'instrumentalness_%',
         'liveness_%',
         'speechiness_%'
      ]

      # ลบเครื่องหมายคอมมาแล้วแปลงเป็นตัวเลข
      # errors='coerce' = ค่าที่แปลงไม่ได้จะกลายเป็น NaN แทนที่จะ error
      for col in numeric_cols:
         if col in data_frame.columns:
            data_frame[col] = pd.to_numeric(data_frame[col].astype(str).str.replace(',', ''), errors='coerce')

      # streams คือค่าหลักของงานนี้ แถวที่ streams เสียหาย (มี 1 แถว) ตัดทิ้ง
      # reset_index เพื่อให้เลขแถวต่อเนื่อง 0..n-1 (จำเป็นเพราะ x_scaled ใช้ลำดับแถวเดียวกัน)
      return data_frame.dropna(subset=['streams']).reset_index(drop=True)


# ==========================================================================
# 2) Store: เก็บฟีเจอร์ + Standardize + หาค่า k
# ==========================================================================
class Store:
   """เก็บทุกอย่างของ 1 โมเดล: DataFrame, ฟีเจอร์ที่ใช้, ข้อมูลที่ Standardize แล้ว, ผลประเมินค่า k"""

   def __init__(self, data_frame: pd.DataFrame, features: list[str], name: str, k_range: range):
      self.data_frame = data_frame
      self.features = features
      self.name = name                                    # ชื่อโมเดล เช่น "pop", "audio"
      self.name_cluster = name + "_cluster"               # คอลัมน์เลขกลุ่ม เช่น pop_cluster
      self.name_cluster_label = name + "_cluster_label"   # คอลัมน์ชื่อกลุ่ม
      self.k_range = k_range                              # ช่วง k ที่จะทดลอง
      self.k_cluster = None                               # k ที่เลือกใช้จริง
      self.x_scaled = None                                # ข้อมูลหลัง Standardize
      self.evaluation = {
         "elbow": [],
         "silhouette": []
      }

   def find_x_scaled(self, log: bool = False):
      """Feature Engineering: (ทางเลือก) log1p แล้ว Standardize

      log1p = log(1 + x) ใช้กับข้อมูลความนิยมที่เบ้ขวามาก (เพลงดังสุดๆ ไม่กี่เพลงมีค่าสูงกว่าเพลงอื่นหลายเท่า)
      ใช้ 1 + x เพราะบางค่าเป็น 0 (หา log(0) ไม่ได้)

      Standardize: z = (x - ค่าเฉลี่ย) / ส่วนเบี่ยงเบนมาตรฐาน ให้ทุกฟีเจอร์มีเฉลี่ย 0 และ SD = 1
      เพราะ K-means วัดความคล้ายด้วยระยะทาง ถ้าสเกลต่างกัน (เช่น streams เป็นพันล้าน
      เทียบกับ valence 0-100) ฟีเจอร์ที่ตัวเลขใหญ่จะครอบงำการจัดกลุ่ม
      """
      x = self.data_frame[self.features].astype(float)
      if log:
         x = np.log1p(x)
      self.x_scaled = StandardScaler().fit_transform(x)

   def set_k_cluster(self, k_cluster: int):
      self.k_cluster = k_cluster

   def find_evaluation(self):
      """ทดลอง K-means ทุกค่า k ในช่วง แล้วเก็บ inertia (Elbow) และ Silhouette ไว้เลือก k

      - inertia: ผลรวมระยะห่างภายในกลุ่ม ยิ่งต่ำยิ่งกระชับ (ลดลงเรื่อยๆ เมื่อ k เพิ่ม ดูจุด "ศอก")
      - silhouette: -1 ถึง 1 ยิ่งใกล้ 1 ยิ่งแยกกลุ่มชัด
      """
      self.evaluation = {"elbow": [], "silhouette": []}
      print(f"--- หาค่า k ของ {self.name} ---")
      for k in self.k_range:
         km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
         labels = km.fit_predict(self.x_scaled)
         self.evaluation["elbow"].append(km.inertia_)
         self.evaluation["silhouette"].append(silhouette_score(self.x_scaled, labels))
         print(f"k={k}  inertia={self.evaluation['elbow'][-1]:10.1f}  "
               f"silhouette={self.evaluation['silhouette'][-1]:.3f}")


# ==========================================================================
# 3) Cluster: จัดกลุ่ม + โปรไฟล์ + ตั้งชื่อ
# ==========================================================================
class Cluster:
   def __init__(self, store: Store):
      self.store = store
      self.k_means = None
      self.label = None          # dict {เลขกลุ่ม: ชื่อกลุ่ม}
      self.silhouette = None     # silhouette ของ k ที่เลือกใช้

   def fit(self):
      """จัดกลุ่มด้วย K-means ตาม k ที่เลือก แล้วเพิ่มคอลัมน์เลขกลุ่มลงใน DataFrame

      n_init=10 : รัน 10 ครั้งด้วยจุดเริ่มต้นต่างกัน เก็บผลที่ดีที่สุด (K-means ไวต่อจุดเริ่มต้น)
      """
      self.k_means = KMeans(n_clusters=self.store.k_cluster, random_state=RANDOM_STATE, n_init=10)
      self.store.data_frame[self.store.name_cluster] = self.k_means.fit_predict(self.store.x_scaled)
      self.silhouette = silhouette_score(self.store.x_scaled, self.store.data_frame[self.store.name_cluster])
      print(f"\n{self.store.name}: k={self.store.k_cluster}  silhouette={self.silhouette:.3f}")

   def profile(self, stat: str = "mean") -> pd.DataFrame:
      """โปรไฟล์ของแต่ละกลุ่ม (stat = "mean" ค่าเฉลี่ย หรือ "median" มัธยฐาน)

      ข้อมูลความนิยมเบ้มาก ค่าเฉลี่ยถูกเพลงดังสุดๆ ดึงขึ้น จึงนิยมใช้มัธยฐานตีความกลุ่ม
      """
      g = self.store.data_frame.groupby(self.store.name_cluster)
      prof = g[self.store.features].agg(stat)
      prof["n_tracks"] = g.size()
      prof["median_year"] = g["released_year"].median()
      return prof

   def set_label(self, labels: dict[int, str]):
      """ตั้งชื่อกลุ่ม แล้วเพิ่มคอลัมน์ชื่อกลุ่มลงใน DataFrame (ตรวจว่าชื่อครบตามจำนวน k)"""
      if set(labels) != set(range(self.store.k_cluster)):
         raise ValueError(f"labels ต้องมี key เป็น 0..{self.store.k_cluster - 1} "
                          f"แต่ได้ {sorted(labels)}")
      self.label = labels
      self.store.data_frame[self.store.name_cluster_label] = self.store.data_frame[self.store.name_cluster].map(labels)

# ==========================================================================
# 4) AutoLabel: ตั้งชื่อกลุ่มอัตโนมัติจากโปรไฟล์
#    เลขกลุ่มของ K-means สลับได้เมื่อเปลี่ยนข้อมูล/ฟีเจอร์/เวอร์ชัน ถ้าเขียนตายตัวว่า
#    {0: ..., 1: ...} ชื่ออาจติดผิดกลุ่มโดยไม่มี error เตือน จึงให้ตั้งชื่อจากค่าในโปรไฟล์แทน
#    (ชื่อกลุ่มเป็นการตีความของผู้วิเคราะห์ ไม่ใช่ผลจากโมเดลโดยตรง และรองรับเฉพาะ k = 3)
# ==========================================================================
class AutoLabel:
   @staticmethod
   def popularity(profile: pd.DataFrame) -> dict[int, str]:
      """
      cluster : streams สูงสุด = Megahits
      cluster : ในที่เหลือ ติดชาร์ต Spotify มากกว่า = Top Charting
      cluster : อีกกลุ่ม = Moderate
      """
      order = profile["streams"].sort_values(ascending=False).index.tolist()
      top, rest = order[0], order[1:]
      charting = profile.loc[rest, "in_spotify_charts"].idxmax()
      moderate = [c for c in rest if c != charting][0]
      return {int(top): LABEL_MEGA, 
              int(charting): LABEL_CHARTING, 
              int(moderate): LABEL_MODERATE}

   @staticmethod
   def audio(profile: pd.DataFrame) -> dict[int, str]:
      """
      cluster : speechiness สูงสุด = Rap
      cluster : ในที่เหลือ acousticness สูงกว่า = Acoustic 
      cluster : อีกกลุ่ม = Upbeat
      """
      rap = profile["speechiness_%"].idxmax()
      rest = [c for c in profile.index if c != rap]
      acoustic = profile.loc[rest, "acousticness_%"].idxmax()
      upbeat = [c for c in rest if c != acoustic][0]
      return {int(rap): "Rap\nHip-Hop\nTalk-Heavy",
              int(acoustic): "Acoustic\nChill\nMelancholic",
              int(upbeat): "Upbeat\nDance\nParty"}


# ==========================================================================
# 5) Chart: วาดกราฟ
# ==========================================================================
class Chart:
   @staticmethod
   def plot_evaluation(chart_directory: str, store: Store) -> None:
      """กราฟ Elbow และ Silhouette ตามค่า k ใช้ประกอบการเลือก k"""
      _, axes = plt.subplots(1, 2, figsize=(16, 6))

      axes[0].plot(store.k_range, store.evaluation["elbow"], 'o-', color='#1f77b4', linewidth=1.5, markersize=6)
      axes[0].set_title(f'Elbow Method of {store.name}', fontsize=12)
      axes[0].set_xlabel('number of cluster (k)', fontsize=10)
      axes[0].set_ylabel('inertia', fontsize=10)
      axes[0].set_xticks(list(store.k_range))
      axes[1].plot(store.k_range, store.evaluation["silhouette"], 'o-', color='#2ca02c', linewidth=1.5, markersize=6)
      axes[1].set_title(f'Silhouette Method of {store.name}', fontsize=12)
      axes[1].set_xlabel('number of cluster (k)', fontsize=10)
      axes[1].set_ylabel('silhouette score', fontsize=10)
      axes[1].set_xticks(list(store.k_range))

      plt.tight_layout()
      plt.savefig(f'{chart_directory}/evaluation-{store.name}-{int(time.time())}.png', dpi=300)
      plt.close()

   @staticmethod
   def plot_cluster(chart_directory: str, store: Store, cluster: Cluster) -> None:
      """กราฟซ้าย: scatter ของกลุ่มบน PCA 2 แกน | กราฟขวา: จำนวนเพลงในแต่ละกลุ่ม"""
      from matplotlib.lines import Line2D

      _, axes = plt.subplots(1, 2, figsize=(14, 5))

      # PCA ลดข้อมูลหลายมิติเหลือ 2 มิติเพื่อวาดกราฟเท่านั้น (ไม่ได้ใช้จัดกลุ่ม)
      pca = PCA(n_components=2, random_state=RANDOM_STATE)
      coords = pca.fit_transform(store.x_scaled)
      centers = pca.transform(cluster.k_means.cluster_centers_)

      labels = store.data_frame[store.name_cluster].values

      # สีของแต่ละ cluster (ใช้ร่วมกันทั้ง scatter และ bar)
      cmap = plt.cm.tab10
      norm = plt.Normalize(vmin=labels.min(), vmax=labels.max())
      cluster_ids = sorted(cluster.label.keys())
      cluster_colors = {cid: cmap(norm(cid)) for cid in cluster_ids}

      axes[0].scatter(coords[:, 0], coords[:, 1], c=labels, cmap=cmap, norm=norm, s=18, alpha=0.7)
      axes[0].scatter(centers[:, 0], centers[:, 1], c='black', marker='X', s=200,
                      edgecolors='white', linewidths=1.5)   # จุดศูนย์กลางกลุ่ม (centroid)
      axes[0].set_title(f'K-Means Clusters (k={store.k_cluster}) - PCA 2D', fontsize=12)
      axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
      axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')

      # legend สร้างตามเลขกลุ่ม เพื่อให้ชื่อตรงกับสีเสมอ
      handles = [Line2D([0], [0], marker='o', linestyle='', markersize=8, color=cluster_colors[cid],
                        label=cluster.label[cid].replace("\n", " ")) for cid in cluster_ids]
      handles.append(Line2D([0], [0], marker='X', linestyle='', markersize=10, color='black', label='centroid'))
      axes[0].legend(handles=handles)

      counts = store.data_frame[store.name_cluster].value_counts().sort_index()
      bar_names = [cluster.label[i] for i in counts.index]
      bar_colors = [cluster_colors[i] for i in counts.index]
      bars = axes[1].bar(bar_names, counts.values, color=bar_colors)
      axes[1].bar_label(bars)
      axes[1].set_title('Number of songs per cluster', fontsize=12)
      axes[1].set_xlabel('cluster')
      axes[1].set_ylabel('music')

      plt.tight_layout()
      plt.savefig(f"{chart_directory}/cluster-{store.name}-{int(time.time())}.png", dpi=300)
      plt.close()


# ==========================================================================
# 6) แนะนำเพลง (ใช้หลังจัดกลุ่มเสร็จ)
# ==========================================================================
# ลงทะเบียนโมเดลที่จัดกลุ่มแล้วไว้ที่นี่ (ทำใน main) recommend() จะดึงไปใช้ตามชื่อ "audio" / "pop"
MODELS: dict[str, tuple[Store, Cluster]] = {}

def recommend(song: str, model: str, count: int, max_per_artist: int | None = None,
              verbose: bool = True) -> pd.DataFrame:
   """แนะนำเพลงจากกลุ่มเดียวกับเพลงเริ่มต้น เรียงจากใกล้ที่สุดไปไกล

   song           : ชื่อเพลงเริ่มต้น (ไม่สนตัวพิมพ์เล็ก/ใหญ่)
   model          : "audio" = เพลงที่เสียงคล้ายกัน | "pop" = เพลงที่ระดับความนิยมใกล้กัน
   count          : จำนวนเพลงที่ต้องการ
   max_per_artist : (ไม่บังคับ) จำกัดจำนวนเพลงต่อศิลปิน เช่น 1 = ไม่ซ้ำศิลปิน
   verbose        : True = พิมพ์หัวข้อบอกเพลงเริ่มต้นและกลุ่ม

   ขั้นตอน:
     1) หาเพลงเริ่มต้นและดูว่าอยู่ cluster ไหนของ model ที่เลือก
     2) เอาเพลงอื่นใน cluster เดียวกันมาเป็นผู้สมัคร
     3) เรียงด้วยระยะห่าง (Euclidean) ในปริภูมิฟีเจอร์ที่ Standardize แล้วของ model นั้น
     4) คืน count เพลงที่ใกล้ที่สุด
   """
   key = model.strip().lower()
   if key not in MODELS:
      raise ValueError(f"model ต้องเป็นหนึ่งใน {list(MODELS)} แต่ได้ '{model}'")
   store, cluster = MODELS[key]
   df = store.data_frame
   col = store.name_cluster

   # 1) หาเพลงเริ่มต้น (ชื่อซ้ำ -> เลือกตัวที่ streams สูงสุด)
   titles = df["track_name"].astype(str)
   hit = df[titles.str.lower() == song.strip().lower()]
   if hit.empty:
      close = difflib.get_close_matches(song, titles.tolist(), n=5, cutoff=0.5)
      raise ValueError(f"ไม่พบเพลง '{song}' ชื่อที่ใกล้เคียง: {close}")
   seed_idx = int(hit["streams"].idxmax())
   seed = df.loc[seed_idx]
   cluster_id = seed[col]
   cluster_name = cluster.label[cluster_id].replace("\n", " ") if cluster.label else f"cluster {cluster_id}"

   # 2-3) ผู้สมัคร = เพลงอื่นในกลุ่มเดียวกัน เรียงตามระยะห่างจากเพลงเริ่มต้น
   X = store.x_scaled                     # แถวของ x_scaled ตรงกับแถวของ DataFrame
   cand = df.assign(distance=np.linalg.norm(X - X[seed_idx], axis=1))
   garbled = (cand["track_name"].astype(str).str.contains(GARBLED_MARK, regex=False)
              | cand["artist(s)_name"].astype(str).str.contains(GARBLED_MARK, regex=False))
   same_title = cand["track_name"].astype(str).str.lower() == str(seed["track_name"]).lower()
   # ตัดเพลงชื่อเพี้ยน และเพลงชื่อเดียวกับเพลงเริ่มต้น (คนละเวอร์ชัน) ออก
   cand = cand[(cand[col] == cluster_id) & ~same_title & ~garbled].sort_values("distance")

   if max_per_artist:
      main_artist = cand["artist(s)_name"].astype(str).str.split(",").str[0].str.strip()
      cand = cand.groupby(main_artist, sort=False).head(max_per_artist)

   # 4) เลือก count เพลงแรก
   out = cand.head(count)
   if len(out) < count:
      print(f"[หมายเหตุ] ได้ {len(out)} เพลง จากที่ขอ {count} (ใน cluster นี้มีผู้สมัครไม่พอ)")

   label_col = store.name_cluster_label
   cols = ["track_name", "artist(s)_name", "released_year", "streams", "distance"]
   if label_col in out.columns:
      cols.insert(3, label_col)
   out = out[cols].copy()
   if label_col in out.columns:
      out[label_col] = out[label_col].str.replace("\n", " ")
   out["streams"] = out["streams"].astype("int64")
   out["distance"] = out["distance"].round(2)     # distance ยิ่งต่ำยิ่งคล้าย (เทียบได้เฉพาะในผลของเพลงเดียวกัน)
   out.insert(0, "rank", range(1, len(out) + 1))

   if verbose:
      print(f"\nเพลงเริ่มต้น: {seed['track_name']} - {seed['artist(s)_name']}  "
            f"| model={key} | กลุ่ม: {cluster_name}")
   return out.reset_index(drop=True)


# ==========================================================================
# 7) Summary: สรุปผลทั้งหมด (คำนวณจากผลที่รันจริง ไม่ได้เขียนตัวเลขตายตัว)
# ==========================================================================
class Summary:
   def __init__(self, data_frame: pd.DataFrame, pop: tuple[Store, Cluster],
                audio: tuple[Store, Cluster]):
      self.df = data_frame
      self.pop_model, self.pop_cluster = pop
      self.audio_model, self.audio_cluster = audio

   @staticmethod
   def _flat(text: str) -> str:
      """เปลี่ยนขึ้นบรรทัดใหม่ในชื่อกลุ่มเป็นช่องว่าง (ชื่อกลุ่มเสียงมี \\n ไว้ใช้ในกราฟ)"""
      return str(text).replace("\n", " ")

   def _best_k_note(self, model: Store) -> str:
      """รายงานว่า k ที่ silhouette สูงสุดคือเท่าไร และต่างจาก k ที่เลือกหรือไม่ (รายงานตามจริง)"""
      scores = model.evaluation["silhouette"]
      best_k = list(model.k_range)[int(np.argmax(scores))]
      best = max(scores)
      note = f"k ที่ silhouette สูงสุดในช่วง {model.k_range.start}-{model.k_range.stop - 1} คือ k={best_k} ({best:.3f})"
      if best_k != model.k_cluster:
         note += (f" แต่เลือกใช้ k={model.k_cluster} เพราะแบ่งกลุ่มที่ตีความได้ละเอียดกว่า "
                  f"โดยยอมให้ silhouette ต่ำลงเล็กน้อย")
      return note

   def build(self, example_song: str = "Perfect") -> str:
      df = self.df
      pop_col, pop_lab = self.pop_model.name_cluster, self.pop_model.name_cluster_label
      aud_col, aud_lab = self.audio_model.name_cluster, self.audio_model.name_cluster_label
      L = []   # เก็บบรรทัดของรายงาน

      L += ["=" * 70, "สรุปผลการจัดกลุ่มและแนะนำเพลง: Top Spotify Songs 2023", "=" * 70]

      # ---- 1) ภาพรวมข้อมูล ----
      L += ["", "1. ภาพรวมข้อมูล",
            f"   - จำนวนเพลงหลังทำความสะอาด: {len(df)} เพลง",
            f"   - ฟีเจอร์ Popularity {len(self.pop_model.features)} ตัว: {', '.join(self.pop_model.features)}",
            f"   - ฟีเจอร์ Audio {len(self.audio_model.features)} ตัว: {', '.join(self.audio_model.features)}"]

      # ---- 2) Popularity ----
      pop_tbl = df.groupby(pop_lab).agg(
         n_tracks=("track_name", "size"),
         streams_median_M=("streams", lambda s: round(s.median() / 1e6)),
         playlist_median=("in_spotify_playlists", "median"),
         charts_median=("in_spotify_charts", "median"),
         median_year=("released_year", "median"),
      ).sort_values("streams_median_M", ascending=False)
      L += ["", f"2. Popularity Model (K-means k={self.pop_model.k_cluster}, log1p + Standardize)",
            f"   Silhouette = {self.pop_cluster.silhouette:.3f}   | {self._best_k_note(self.pop_model)}",
            "   โปรไฟล์กลุ่ม (มัธยฐาน; streams หน่วยล้าน):"]
      L += ["   " + ln for ln in pop_tbl.to_string().split("\n")]

      # ---- 3) Audio ----
      aud_tbl = df.groupby(aud_lab)[self.audio_model.features].mean().round(1)
      aud_tbl.index = [self._flat(i) for i in aud_tbl.index]
      aud_tbl.insert(0, "n_tracks", df.groupby(aud_lab).size().rename(index=self._flat))
      L += ["", f"3. Audio Model (K-means k={self.audio_model.k_cluster}, Standardize)",
            f"   Silhouette = {self.audio_cluster.silhouette:.3f}   | {self._best_k_note(self.audio_model)}",
            "   โปรไฟล์กลุ่ม (ค่าเฉลี่ย):"]
      L += ["   " + ln for ln in aud_tbl.to_string().split("\n")]

      # ---- 4) ไขว้ 2 โมเดล ----
      cross = pd.crosstab(df[aud_lab].map(self._flat), df[pop_lab], normalize="index").mul(100).round(0)
      L += ["", "4. สัดส่วน (%) ของแต่ละกลุ่มเสียง ที่อยู่ในแต่ละกลุ่มความนิยม"]
      L += ["   " + ln for ln in cross.to_string().split("\n")]

      # ---- 5) ข้อค้นพบ (คำนวณจากผลจริง) ----
      mega = cross[LABEL_MEGA]
      newest = pop_tbl["median_year"].idxmax()
      better = "Popularity" if self.pop_cluster.silhouette > self.audio_cluster.silhouette else "Audio"
      L += ["", "5. ข้อค้นพบ",
            f"   - กลุ่มเสียงที่อยู่ในกลุ่ม {LABEL_MEGA} มากที่สุดคือ '{mega.idxmax()}' ({mega.max():.0f}%) "
            f"และน้อยที่สุดคือ '{mega.idxmin()}' ({mega.min():.0f}%)",
            f"   - กลุ่มความนิยมที่เป็นเพลงใหม่ที่สุดคือ '{newest}' (ปีมัธยฐาน {pop_tbl.loc[newest, 'median_year']:.0f})",
            f"   - โมเดล {better} แยกกลุ่มได้ชัดกว่า (silhouette "
            f"Popularity {self.pop_cluster.silhouette:.3f} เทียบ Audio {self.audio_cluster.silhouette:.3f})",
            "   - หมายเหตุ: ข้อค้นพบข้างต้นเป็นแนวโน้มจากการนับสัดส่วน ไม่ได้ทดสอบนัยสำคัญทางสถิติ"]

      # ---- 6) ตัวอย่างการแนะนำเพลง ----
      if example_song.lower() not in df["track_name"].astype(str).str.lower().values:
         example_song = df.loc[df["streams"].idxmax(), "track_name"]   # ถ้าไม่พบ ใช้เพลงที่ streams สูงสุด
      L += ["", f"6. ตัวอย่างการแนะนำเพลง จากเพลง '{example_song}' (recommend(song, model, count))"]
      for m, desc in [("audio", "เสียงคล้ายกัน"), ("pop", "ความนิยมใกล้เคียงกัน")]:
         rec = recommend(example_song, m, 5, verbose=False)
         L.append(f"   model={m} ({desc}):")
         L += [f"     {r['rank']}. {r['track_name']} - {r['artist(s)_name']}" for _, r in rec.iterrows()]

      # ---- 7) ข้อจำกัด ----
      L += ["", "7. ข้อจำกัดและแนวทางต่อยอด",
            "   - ชุดข้อมูลไม่มี genre และพฤติกรรมผู้ใช้ จึงวัดความแม่นยำของการแนะนำไม่ได้ (ไม่มี ground truth)",
            "   - ชื่อกลุ่มตั้งจากโปรไฟล์โดยผู้วิเคราะห์ ไม่ใช่ผลจากโมเดลโดยตรง",
            "   - Audio silhouette ต่ำ ขอบเขตกลุ่มซ้อนกัน เพลงที่อยู่ใกล้เส้นแบ่งอาจอยู่คนละกลุ่ม",
            "   - ข้อมูลมีเฉพาะเพลงที่ฮิตอยู่แล้วในปี 2023 จึงไม่ครอบคลุมเพลงทั่วไป",
            "   - ต่อยอด: เพิ่ม genre (เช่น Spotify API), รวมผล 2 โมเดล, ใช้ KNN หาเพลงใกล้เคียงทั้งชุด", ""]
      return "\n".join(L)

   def save(self, text: str, output_directory: str):
      """บันทึกรายงานสรุปและข้อมูลรายเพลงพร้อมชื่อกลุ่ม"""
      os.makedirs(output_directory, exist_ok=True)
      with open(f"{output_directory}/summary.txt", "w", encoding="utf-8") as f:
         f.write(text)

      out = self.df[["track_name", "artist(s)_name", "released_year", "streams",
                     self.pop_model.name_cluster_label, self.audio_model.name_cluster_label]].copy()
      out[self.audio_model.name_cluster_label] = out[self.audio_model.name_cluster_label].map(self._flat)
      # utf-8-sig เพื่อให้เปิดใน Excel แล้วภาษาไทย/อักขระพิเศษไม่เพี้ยน
      out.to_csv(f"{output_directory}/spotify_2023_clustered.csv", index=False, encoding="utf-8-sig")


# ==========================================================================
# MAIN
# ==========================================================================
if __name__ == "__main__":
   CHART_DIRECTORY = "chart"
   OUTPUT_DIRECTORY = "output"
   PATH = os.environ.get("SPOTIFY_CSV", "spotify_2023.csv")
   os.makedirs(CHART_DIRECTORY, exist_ok=True)

   # ----------------------------------------------------
   #  DATA Preprocessing: โหลดและทำความสะอาด
   # ----------------------------------------------------

   data_raw = DataProcessing.load_data(PATH)
   data_frame = DataProcessing.clean_data(data_raw)


   # ----------------------------------------------------
   #  Popularity จัดกลุ่มตามความนิยม
   # ----------------------------------------------------
   #  Data Store:
   #  1. สร้าง Object เก็บ name, k_range, data_frame, features
   #  2. log=True: ข้อมูลความนิยมเบ้มาก ต้อง log1p ก่อน Standardize
   #  3. รัน KMeans สำหรับทุกค่า k ใน k_range คำนวณ inertia (Elbow) และ Silhouette Score 
   #  4. บันทึกกราฟเปรียบเทียบ
   #  5. กำหนดจำนวน cluster

   pop_store = Store(
      name = "pop",
      k_range = range(2, 9),
      data_frame = data_frame,
      features = [
         'streams',
         'in_spotify_playlists',
         'in_spotify_charts',
      ], 
   )

   pop_store.find_x_scaled(log=True)
   pop_store.find_evaluation()
   Chart.plot_evaluation(CHART_DIRECTORY, pop_store)
   pop_store.set_k_cluster(k_cluster = 3)

   #  Cluster:
   #  1. สร้าง model เอา data store มาใส่
   #  2. รัน KMeans ด้วยค่า k ที่เลือก เพิ่มคอลัมน์ cluster ลงใน DataFrame และพิมพ์จำนวนเพลงในแต่ละ cluster
   #  3. ตั้งชื่อจากโปรไฟล์ เพราะข้อมูลความนิยมเบ้ (ใช้มัธยฐาน features ในแต่ละ cluster เพื่อ Identify cluster )
   #  4. บันทึกกราฟเปรียบเทียบ

   pop_cluster = Cluster(pop_store)
   pop_cluster.fit()
   pop_labels = AutoLabel.popularity(pop_cluster.profile("median"))
   pop_cluster.set_label(pop_labels)
   print("ชื่อกลุ่มความนิยม:", pop_labels)
   Chart.plot_cluster(CHART_DIRECTORY, pop_store, pop_cluster)


   # ----------------------------------------------------
   #  Audio Clustering: จัดกลุ่มตามลักษณะเสียง
   # ----------------------------------------------------
   #  Data Store:
   #  1. สร้าง Object เก็บ name, k_range, data_frame, features
   #  2. ทำ Standardize
   #  3. รัน KMeans สำหรับทุกค่า k ใน k_range คำนวณ inertia (Elbow) และ Silhouette Score 
   #  4. บันทึกกราฟเปรียบเทียบ
   #  5. กำหนดจำนวน cluster

   audio_store = Store(
      name = "audio",
      k_range = range(2,9),
      data_frame = data_frame,
      features = [
         'bpm',
         'danceability_%',
         'valence_%',
         'energy_%',
         'acousticness_%',
         'instrumentalness_%',
         'liveness_%',
         'speechiness_%'
      ]
   )

   audio_store.find_x_scaled()
   audio_store.find_evaluation()
   Chart.plot_evaluation(CHART_DIRECTORY, audio_store)
   audio_store.set_k_cluster(k_cluster = 3)

   #  Cluster:
   #  1. สร้าง model เอา data store มาใส่
   #  2. รัน KMeans ด้วยค่า k ที่เลือก เพิ่มคอลัมน์ cluster ลงใน DataFrame และพิมพ์จำนวนเพลงในแต่ละ cluster
   #  3. ตั้งชื่อจากโปรไฟล์ (ใช้ค่าเฉลี่ย ของ features ในแต่ละ cluster เพื่อ Identify cluster)
   #  4. บันทึกกราฟเปรียบเทียบ

   audio_cluster = Cluster(audio_store)
   audio_cluster.fit()
   audio_labels = AutoLabel.audio(audio_cluster.profile("mean"))
   audio_cluster.set_label(audio_labels)
   print("ชื่อกลุ่มเสียง:", {k: v.replace("\n", " ") for k, v in audio_labels.items()})
   Chart.plot_cluster(CHART_DIRECTORY, audio_store, audio_cluster)



   # ----------------------------------------------------
   # แนะนำเพลง
   # ----------------------------------------------------
   # ลงทะเบียนโมเดลที่จัดกลุ่มแล้ว ให้ recommend() ใช้
   
   MODELS["audio"] = (audio_store, audio_cluster)
   MODELS["pop"] = (pop_store, pop_cluster)
   pd.set_option("display.width", 220)
   pd.set_option("display.max_columns", 20)
   print("\n##### แนะนำจากเสียงที่คล้ายกัน #####")
   print(recommend("Perfect", "audio", 5).to_string(index=False))
   print("\n##### แนะนำจากระดับความนิยมที่ใกล้กัน #####")
   print(recommend("Perfect", "pop", 5).to_string(index=False))
   print("\n##### แนะนำจากเสียง ไม่ซ้ำศิลปิน #####")
   print(recommend("Blinding Lights", "audio", 5, max_per_artist=1).to_string(index=False))



   # ----------------------------------------------------
   # สรุปผล
   # ----------------------------------------------------

   SUMMARY = Summary(data_frame, pop=(pop_store, pop_cluster), audio=(audio_store, audio_cluster))
   SUMMARY_TEXT = SUMMARY.build(example_song="Perfect")
   print("\n" + SUMMARY_TEXT)
   SUMMARY.save(SUMMARY_TEXT, OUTPUT_DIRECTORY)
   print(f"บันทึกแล้ว: {OUTPUT_DIRECTORY}/summary.txt และ {OUTPUT_DIRECTORY}/spotify_2023_clustered.csv")