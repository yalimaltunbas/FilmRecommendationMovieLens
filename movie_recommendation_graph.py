"""
Film Öneri Sistemi - Graph Theory Projesi
MovieLens Veri Seti ile Bipartite ve Monopartite Graf Analizi
Personalized PageRank ile Öneri Algoritması
"""

import pandas as pd
import numpy as np
import networkx as nx
from networkx.algorithms import bipartite
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. VERİ YÜKLEME
# =============================================================================

def load_movielens_data(data_path='ml-latest-small'):
    """MovieLens veri setini yükler"""
    
    # Ratings verisi
    ratings = pd.read_csv(f'{data_path}/ratings.csv')
    
    # Film bilgileri
    movies = pd.read_csv(f'{data_path}/movies.csv')
    
    print("=" * 60)
    print("VERİ SETİ BİLGİLERİ")
    print("=" * 60)
    print(f"Toplam rating sayısı: {len(ratings):,}")
    print(f"Benzersiz kullanıcı sayısı: {ratings['userId'].nunique():,}")
    print(f"Benzersiz film sayısı: {ratings['movieId'].nunique():,}")
    print(f"Rating aralığı: {ratings['rating'].min()} - {ratings['rating'].max()}")
    print(f"Ortalama rating: {ratings['rating'].mean():.2f}")
    
    return ratings, movies

# =============================================================================
# 2. BİPARTİTE GRAF OLUŞTURMA
# =============================================================================

def create_bipartite_graph(ratings, min_rating=3.5):
    """
    Kullanıcı-Film bipartite grafı oluşturur
    
    Parameters:
    - ratings: Rating DataFrame
    - min_rating: Sadece bu değer ve üzeri ratingler kenar olarak eklenir
    """
    
    B = nx.Graph()
    
    # Düğümleri ekle (bipartite özelliği ile)
    users = ratings['userId'].unique()
    movies = ratings['movieId'].unique()
    
    # Kullanıcı düğümleri (bipartite=0)
    B.add_nodes_from([f"u_{u}" for u in users], bipartite=0, node_type='user')
    
    # Film düğümleri (bipartite=1)
    B.add_nodes_from([f"m_{m}" for m in movies], bipartite=1, node_type='movie')
    
    # Kenarları ekle (sadece yüksek ratingler)
    high_ratings = ratings[ratings['rating'] >= min_rating]
    
    for _, row in high_ratings.iterrows():
        user_node = f"u_{int(row['userId'])}"
        movie_node = f"m_{int(row['movieId'])}"
        B.add_edge(user_node, movie_node, weight=row['rating'])
    
    print("\n" + "=" * 60)
    print("BİPARTİTE GRAF BİLGİLERİ")
    print("=" * 60)
    print(f"Toplam düğüm sayısı: {B.number_of_nodes():,}")
    print(f"  - Kullanıcı düğümleri: {len(users):,}")
    print(f"  - Film düğümleri: {len(movies):,}")
    print(f"Toplam kenar sayısı: {B.number_of_edges():,}")
    print(f"Graf yoğunluğu (density): {nx.density(B):.6f}")
    print(f"Bipartite kontrol: {bipartite.is_bipartite(B)}")
    
    return B

# =============================================================================
# 3. MONOPARTİTE PROJEKSİYON (Film-Film Benzerlik Grafı)
# =============================================================================

def create_movie_projection(B, min_common_users=5):
    """
    Bipartite graftan film-film monopartite projeksiyon oluşturur
    İki film arasındaki kenar ağırlığı = ortak kullanıcı sayısı
    """
    
    # Film düğümlerini al
    movie_nodes = {n for n, d in B.nodes(data=True) if d.get('bipartite') == 1}
    
    # Weighted projeksiyon
    movie_graph = bipartite.weighted_projected_graph(B, movie_nodes)
    
    # Düşük ağırlıklı kenarları filtrele (gürültüyü azalt)
    edges_to_remove = [(u, v) for u, v, d in movie_graph.edges(data=True) 
                       if d['weight'] < min_common_users]
    movie_graph.remove_edges_from(edges_to_remove)
    
    # İzole düğümleri kaldır
    isolated = list(nx.isolates(movie_graph))
    movie_graph.remove_nodes_from(isolated)
    
    print("\n" + "=" * 60)
    print("MONOPARTİTE GRAF (Film-Film Benzerliği)")
    print("=" * 60)
    print(f"Düğüm sayısı (filmler): {movie_graph.number_of_nodes():,}")
    print(f"Kenar sayısı (benzerlik bağlantıları): {movie_graph.number_of_edges():,}")
    print(f"Graf yoğunluğu: {nx.density(movie_graph):.6f}")
    print(f"Ortalama derece: {sum(dict(movie_graph.degree()).values()) / movie_graph.number_of_nodes():.2f}")
    
    return movie_graph

# =============================================================================
# 4. GRAF METRİKLERİ
# =============================================================================

def calculate_graph_metrics(G, graph_name="Graf"):
    """Graf için temel metrikleri hesaplar"""
    
    print("\n" + "=" * 60)
    print(f"{graph_name} - DETAYLI METRİKLER")
    print("=" * 60)
    
    # Temel metrikler
    print(f"\n[Temel Metrikler]")
    print(f"Düğüm sayısı: {G.number_of_nodes():,}")
    print(f"Kenar sayısı: {G.number_of_edges():,}")
    print(f"Yoğunluk (Density): {nx.density(G):.6f}")
    
    # Derece dağılımı
    degrees = [d for n, d in G.degree()]
    print(f"\n[Derece Dağılımı]")
    print(f"Ortalama derece: {np.mean(degrees):.2f}")
    print(f"Medyan derece: {np.median(degrees):.2f}")
    print(f"Maksimum derece: {max(degrees)}")
    print(f"Minimum derece: {min(degrees)}")
    print(f"Standart sapma: {np.std(degrees):.2f}")
    
    # Bağlantılılık
    print(f"\n[Bağlantılılık]")
    if nx.is_connected(G):
        print("Graf bağlantılı (connected): Evet")
        print(f"Çap (diameter): {nx.diameter(G)}")
        print(f"Ortalama en kısa yol: {nx.average_shortest_path_length(G):.2f}")
    else:
        components = list(nx.connected_components(G))
        print(f"Graf bağlantılı (connected): Hayır")
        print(f"Bağlantılı bileşen sayısı: {len(components)}")
        largest_cc = max(components, key=len)
        print(f"En büyük bileşen düğüm sayısı: {len(largest_cc):,}")
        
        # En büyük bileşen için metrikler
        largest_subgraph = G.subgraph(largest_cc).copy()
        if len(largest_cc) < 1000:  # Hesaplama maliyeti için sınır
            print(f"En büyük bileşen çapı: {nx.diameter(largest_subgraph)}")
            print(f"En büyük bileşen ort. yol: {nx.average_shortest_path_length(largest_subgraph):.2f}")
    
    # Kümeleme katsayısı
    print(f"\n[Kümeleme Katsayısı]")
    avg_clustering = nx.average_clustering(G)
    print(f"Ortalama kümeleme katsayısı: {avg_clustering:.4f}")
    
    # En yüksek dereceli düğümler
    print(f"\n[En Yüksek Dereceli 10 Düğüm]")
    degree_dict = dict(G.degree())
    top_nodes = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)[:10]
    for node, degree in top_nodes:
        print(f"  {node}: {degree}")
    
    return {
        'nodes': G.number_of_nodes(),
        'edges': G.number_of_edges(),
        'density': nx.density(G),
        'avg_degree': np.mean(degrees),
        'avg_clustering': avg_clustering
    }

# =============================================================================
# 5. PERSONALIZED PAGERANK İLE ÖNERİ SİSTEMİ
# =============================================================================

def personalized_pagerank_recommendation(G, seed_movies, movies_df, top_n=10, alpha=0.85):
    """
    Personalized PageRank ile film önerisi yapar
    
    Parameters:
    - G: Film-film benzerlik grafı (monopartite)
    - seed_movies: Kullanıcının beğendiği film ID'leri listesi
    - movies_df: Film bilgileri DataFrame
    - top_n: Kaç öneri döndürülecek
    - alpha: Damping factor (0.85 standart)
    """
    
    # Seed düğümlerini hazırla
    personalization = {}
    seed_nodes = [f"m_{mid}" for mid in seed_movies if f"m_{mid}" in G.nodes()]
    
    if not seed_nodes:
        print("Uyarı: Girilen filmler grafta bulunamadı!")
        return []
    
    # Personalization vektörü (seed filmlerine eşit ağırlık)
    for node in G.nodes():
        personalization[node] = 1.0 / len(seed_nodes) if node in seed_nodes else 0.0
    
    # Personalized PageRank hesapla
    ppr_scores = nx.pagerank(G, alpha=alpha, personalization=personalization, weight='weight')
    
    # Seed filmleri hariç tut ve sırala
    recommendations = [(node, score) for node, score in ppr_scores.items() 
                       if node not in seed_nodes]
    recommendations.sort(key=lambda x: x[1], reverse=True)
    
    # Film isimlerini ekle
    print("\n" + "=" * 60)
    print("PERSONALIZED PAGERANK ÖNERİLERİ")
    print("=" * 60)
    print(f"Seed filmler: {seed_movies}")
    print(f"Alpha (damping factor): {alpha}")
    print(f"\nÖnerilen Filmler:")
    print("-" * 60)
    
    result = []
    for i, (node, score) in enumerate(recommendations[:top_n], 1):
        movie_id = int(node.split('_')[1])
        movie_info = movies_df[movies_df['movieId'] == movie_id]
        if not movie_info.empty:
            title = movie_info['title'].values[0]
            genres = movie_info['genres'].values[0]
            print(f"{i:2}. {title}")
            print(f"    Genres: {genres}")
            print(f"    PPR Score: {score:.6f}")
            result.append({'rank': i, 'movieId': movie_id, 'title': title, 
                          'genres': genres, 'ppr_score': score})
    
    return result

# =============================================================================
# 6. LINK PREDICTION İLE ÖNERİ
# =============================================================================

def link_prediction_scores(G, seed_movie, movies_df, top_n=10):
    """
    Link Prediction metrikleri ile film önerisi
    - Common Neighbors
    - Jaccard Coefficient
    - Adamic-Adar Index
    """
    
    seed_node = f"m_{seed_movie}"
    
    if seed_node not in G.nodes():
        print(f"Film {seed_movie} grafta bulunamadı!")
        return None
    
    # Seed filmin komşuları
    seed_neighbors = set(G.neighbors(seed_node))
    
    # Komşu olmayan düğümler için skorları hesapla
    non_neighbors = set(G.nodes()) - seed_neighbors - {seed_node}
    
    scores = []
    for node in non_neighbors:
        node_neighbors = set(G.neighbors(node))
        
        # Common Neighbors
        common = len(seed_neighbors & node_neighbors)
        
        # Jaccard Coefficient
        union = len(seed_neighbors | node_neighbors)
        jaccard = common / union if union > 0 else 0
        
        # Adamic-Adar Index
        adamic_adar = sum(1 / np.log(G.degree(w)) 
                         for w in seed_neighbors & node_neighbors 
                         if G.degree(w) > 1)
        
        scores.append({
            'node': node,
            'common_neighbors': common,
            'jaccard': jaccard,
            'adamic_adar': adamic_adar
        })
    
    # Adamic-Adar'a göre sırala
    scores.sort(key=lambda x: x['adamic_adar'], reverse=True)
    
    print("\n" + "=" * 60)
    print("LINK PREDICTION ÖNERİLERİ")
    print("=" * 60)
    seed_info = movies_df[movies_df['movieId'] == seed_movie]
    if not seed_info.empty:
        print(f"Seed Film: {seed_info['title'].values[0]}")
    print(f"\nTop {top_n} Öneriler (Adamic-Adar Index'e göre):")
    print("-" * 60)
    
    for i, item in enumerate(scores[:top_n], 1):
        movie_id = int(item['node'].split('_')[1])
        movie_info = movies_df[movies_df['movieId'] == movie_id]
        if not movie_info.empty:
            title = movie_info['title'].values[0]
            print(f"{i:2}. {title}")
            print(f"    Common Neighbors: {item['common_neighbors']}, "
                  f"Jaccard: {item['jaccard']:.4f}, "
                  f"Adamic-Adar: {item['adamic_adar']:.4f}")
    
    return scores[:top_n]

# =============================================================================
# 7. GEPHİ EXPORT
# =============================================================================

def export_to_gephi(G, filename, movies_df=None):
    """
    Grafı Gephi için .gexf formatında kaydeder
    """
    
    # Film isimlerini düğüm özelliği olarak ekle
    if movies_df is not None:
        for node in G.nodes():
            if node.startswith('m_'):
                movie_id = int(node.split('_')[1])
                movie_info = movies_df[movies_df['movieId'] == movie_id]
                if not movie_info.empty:
                    G.nodes[node]['label'] = movie_info['title'].values[0]
                    G.nodes[node]['genres'] = movie_info['genres'].values[0]
    
    # Derece bilgisini ekle (Gephi'de boyut için kullanılabilir)
    for node in G.nodes():
        G.nodes[node]['degree'] = G.degree(node)
    
    # GEXF formatında kaydet
    nx.write_gexf(G, filename)
    print(f"\n✓ Graf '{filename}' olarak kaydedildi.")
    print(f"  Gephi'de açmak için: File > Open > {filename}")
    
    return filename

# =============================================================================
# 8. ANA FONKSİYON
# =============================================================================

def main():
    """Ana çalıştırma fonksiyonu"""
    
    print("\n" + "=" * 60)
    print("  GRAPH THEORY PROJESİ - FİLM ÖNERİ SİSTEMİ")
    print("  MovieLens Veri Seti ile Graf Tabanlı Öneri")
    print("=" * 60)
    
    # 1. Veri yükle
    ratings, movies = load_movielens_data('ml-latest-small')
    
    # 2. Bipartite graf oluştur
    bipartite_graph = create_bipartite_graph(ratings, min_rating=3.5)
    
    # 3. Monopartite projeksiyon (film-film)
    movie_graph = create_movie_projection(bipartite_graph, min_common_users=5)
    
    # 4. Graf metriklerini hesapla
    metrics = calculate_graph_metrics(movie_graph, "Film Benzerlik Grafı")
    
    # 5. Popüler filmleri göster (seed seçimi için)
    print("\n" + "=" * 60)
    print("POPÜLER FİLMLER (Yüksek Dereceli)")
    print("=" * 60)
    degree_dict = dict(movie_graph.degree())
    top_movies = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)[:20]
    
    for node, degree in top_movies:
        movie_id = int(node.split('_')[1])
        movie_info = movies[movies['movieId'] == movie_id]
        if not movie_info.empty:
            print(f"ID: {movie_id:5} | Derece: {degree:3} | {movie_info['title'].values[0]}")
    
    # 6. Personalized PageRank ile öneri
    # Örnek: Toy Story (1), Jumanji (2), GoldenEye (10) beğenen biri için
    seed_movies = [1, 2, 10]
    ppr_recommendations = personalized_pagerank_recommendation(
        movie_graph, seed_movies, movies, top_n=10
    )
    
    # 7. Link Prediction ile öneri
    # Örnek: Toy Story (1) için benzer filmler
    link_pred_scores = link_prediction_scores(movie_graph, 1, movies, top_n=10)
    
    # 8. Gephi export
    export_to_gephi(movie_graph, 'movie_similarity_graph.gexf', movies)
    export_to_gephi(bipartite_graph, 'user_movie_bipartite.gexf', movies)
    
    print("\n" + "=" * 60)
    print("PROJE TAMAMLANDI!")
    print("=" * 60)
    print("""
Oluşturulan dosyalar:
1. movie_similarity_graph.gexf - Film benzerlik grafı (Monopartite)
2. user_movie_bipartite.gexf - Kullanıcı-Film grafı (Bipartite)

Gephi'de görselleştirme için:
1. Gephi'yi aç
2. File > Open > .gexf dosyasını seç
3. Layout: ForceAtlas2 veya Fruchterman-Reingold kullan
4. Appearance > Nodes > Size: Degree'ye göre ayarla
5. Appearance > Nodes > Color: Modularity Class'a göre boya
    (Statistics > Modularity çalıştır önce)
    """)
    
    return bipartite_graph, movie_graph, movies, ratings

# =============================================================================
# ÇALIŞTIR
# =============================================================================

if __name__ == "__main__":
    bipartite_graph, movie_graph, movies, ratings = main()
