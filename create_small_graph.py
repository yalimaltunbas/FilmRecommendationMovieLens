"""
Film Öneri Sistemi - Gephi için Optimize Edilmiş Versiyon
Daha az kenar ile daha hızlı görselleştirme
"""

import pandas as pd
import numpy as np
import networkx as nx
from networkx.algorithms import bipartite
import warnings
warnings.filterwarnings('ignore')

def load_data(data_path='ml-latest-small'):
    ratings = pd.read_csv(f'{data_path}/ratings.csv')
    movies = pd.read_csv(f'{data_path}/movies.csv')
    return ratings, movies

def create_small_graph_for_gephi(ratings, movies, min_common_users=20, top_n_movies=500):
    """
    Gephi için optimize edilmiş küçük graf
    - min_common_users: En az bu kadar ortak kullanıcı olan filmler bağlansın
    - top_n_movies: Sadece en popüler N film
    """
    
    print("=" * 60)
    print("GEPHİ İÇİN OPTİMİZE EDİLMİŞ GRAF")
    print("=" * 60)
    
    # Sadece yüksek ratingler (3.5+)
    high_ratings = ratings[ratings['rating'] >= 3.5]
    
    # En çok rating alan filmleri bul
    movie_counts = high_ratings['movieId'].value_counts()
    top_movies = movie_counts.head(top_n_movies).index.tolist()
    
    print(f"Seçilen film sayısı: {len(top_movies)}")
    
    # Sadece bu filmlerle çalış
    filtered_ratings = high_ratings[high_ratings['movieId'].isin(top_movies)]
    
    # Bipartite graf oluştur
    B = nx.Graph()
    
    users = filtered_ratings['userId'].unique()
    
    B.add_nodes_from([f"u_{u}" for u in users], bipartite=0)
    B.add_nodes_from([f"m_{m}" for m in top_movies], bipartite=1)
    
    for _, row in filtered_ratings.iterrows():
        B.add_edge(f"u_{int(row['userId'])}", f"m_{int(row['movieId'])}", weight=row['rating'])
    
    # Monopartite projeksiyon
    movie_nodes = {f"m_{m}" for m in top_movies}
    movie_graph = bipartite.weighted_projected_graph(B, movie_nodes)
    
    # Düşük ağırlıklı kenarları kaldır (DAHA SIKI FİLTRE)
    edges_to_remove = [(u, v) for u, v, d in movie_graph.edges(data=True) 
                       if d['weight'] < min_common_users]
    movie_graph.remove_edges_from(edges_to_remove)
    
    # İzole düğümleri kaldır
    isolated = list(nx.isolates(movie_graph))
    movie_graph.remove_nodes_from(isolated)
    
    print(f"\n[Sonuç Graf]")
    print(f"Düğüm sayısı: {movie_graph.number_of_nodes()}")
    print(f"Kenar sayısı: {movie_graph.number_of_edges()}")
    print(f"Yoğunluk: {nx.density(movie_graph):.4f}")
    
    # Film isimlerini ekle
    for node in movie_graph.nodes():
        movie_id = int(node.split('_')[1])
        movie_info = movies[movies['movieId'] == movie_id]
        if not movie_info.empty:
            title = movie_info['title'].values[0]
            genres = movie_info['genres'].values[0]
            movie_graph.nodes[node]['label'] = title
            movie_graph.nodes[node]['genres'] = genres
        movie_graph.nodes[node]['degree'] = movie_graph.degree(node)
    
    return movie_graph

def main():
    print("\n" + "=" * 60)
    print("  GEPHİ İÇİN KÜÇÜK GRAF OLUŞTURUCU")
    print("=" * 60 + "\n")
    
    ratings, movies = load_data('ml-latest-small')
    
    # =====================================================
    # AYARLAR - Gephi'de takılma olursa bu değerleri artır
    # =====================================================
    MIN_COMMON_USERS = 20  # Ne kadar yüksek = o kadar az kenar
    TOP_N_MOVIES = 300     # Ne kadar düşük = o kadar az düğüm
    # =====================================================
    
    movie_graph = create_small_graph_for_gephi(
        ratings, movies, 
        min_common_users=MIN_COMMON_USERS, 
        top_n_movies=TOP_N_MOVIES
    )
    
    # Kaydet
    output_file = 'movie_graph_small.gexf'
    nx.write_gexf(movie_graph, output_file)
    
    print(f"\n✓ '{output_file}' dosyası oluşturuldu!")
    print("\nGephi'de aç ve ForceAtlas 2 kullan.")
    print("Bu graf çok daha hızlı çalışacak!")

if __name__ == "__main__":
    main()
