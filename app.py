import streamlit as st
import pandas as pd
import pickle
import surprise
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import numpy as np
import math

# Set page configuration
st.set_page_config(page_title="H&M Recommender System", layout="wide")

# Header
header_image = Image.open('Images/h&mBanner.jpeg')
st.image(header_image)

# Creating sidebar comments
st.sidebar.title('H&M Fashion Recommendations')

# Load in appropriate DataFrames, user ratings
articles_df = pd.read_csv('Data/articles.csv.zip', index_col='article_id')
articles_df2 = pd.read_csv('Data/articles.csv.zip')

# Customer data for collabortive filtering
df_customer = pd.read_csv('Data/df_customer.csv', index_col='customer_id')

# Meta data for collabortive filtering
transactions = pd.read_csv('Data/out.zip')

# Meta data for content based
meta_data = pd.read_csv('Data/out_content.zip')

# Import final collab model
collab_model = pickle.load(open('Model/collaborative_model.sav', 'rb'))

# Sample IDs for quick selection
SAMPLE_CUSTOMER_IDS = [
    "002611889659ab1051fc3e4e870f2b603c3aaa902ffe6ab59e83461c76c879dc",
    "ffb72741f3bc3d98855703b55d34e05bc7893a5d6a99a3758cc7fa0cf65ba441",
    "ffc92c3f7b0b302f393c2968b290f6e5c5b5510d1cf1dfd2f8586a5ce3ce8bf4",
    "0008968c0d451dbc5a9968da03196fe20051965edde7413775c4eb3be9abe9c2",
    "0035b1a78ff4c2ae30b91308700c09d257bc8f830acc53574829f391f3e611d4"
]

SAMPLE_ARTICLE_IDS = [
    110065001,
    110065002,
    108775015,
    108775044,
    111565003
]

# Def function using model to return recommendations - collaborative filtering
def customer_article_recommend(customer, n_recs, return_scores=False, return_explanation=False):
    
    have_bought = list(df_customer.loc[customer, 'article_id'])
    not_bought = articles_df.copy()
    not_bought.drop(have_bought, inplace=True)
    not_bought.reset_index(inplace=True)
    not_bought['est_purchase'] = not_bought['article_id'].apply(lambda x: collab_model.predict(customer, x).est)
    not_bought.sort_values(by='est_purchase', ascending=False, inplace=True)
    
    # Store scores for hybrid recommendations and explanations
    if return_scores or return_explanation:
        scores_df = not_bought[['article_id', 'est_purchase']].copy()
        scores_df.set_index('article_id', inplace=True)
        
    not_bought.rename(columns={'prod_name':'Product Name', 'author':'Author',
                              'product_type_name':'Product Type Name', 'product_group_name':'Product Group Name',
                              'index_group_name':'Index Group Name', 'garment_group_name ':'Garment Group Name'}, inplace=True)
    not_bought = not_bought.iloc[:100, :]
    
    # Get purchase history for explanations
    explanations = {}
    if return_explanation:
        # Find products customer has purchased
        past_purchases = articles_df.loc[have_bought].reset_index()
        past_purchases.rename(columns={'prod_name':'Product Name', 'product_type_name':'Product Type Name', 
                                     'product_group_name':'Product Group Name'}, inplace=True)
        
        # Generate explanations for top recommendations
        for idx, article_id in enumerate(not_bought['article_id'].head(n_recs)):
            rec_item = not_bought[not_bought['article_id'] == article_id].iloc[0]
            
            # Find similarities with past purchases
            similar_items = past_purchases[past_purchases['Product Type Name'] == rec_item['Product Type Name']]
            
            if not similar_items.empty:
                explanation = f"Based on your previous purchases of {similar_items['Product Name'].iloc[0]} " + \
                             f"and other {rec_item['Product Type Name']} items in your history."
            else:
                explanation = f"Other customers with similar preferences have purchased this {rec_item['Product Type Name']}."
                
            explanations[article_id] = explanation
    
    if return_scores and return_explanation:
        return not_bought.head(n_recs), scores_df, explanations
    elif return_scores:
        return not_bought.head(n_recs), scores_df
    elif return_explanation:
        return not_bought.head(n_recs), explanations
    
    not_bought.drop(['product_code', 'product_type_no', 'graphical_appearance_no','graphical_appearance_name', 'colour_group_code', 'colour_group_name',
    'perceived_colour_value_id', 'perceived_colour_value_name','perceived_colour_master_id', 'perceived_colour_master_name',
    'department_no', 'department_name', 'index_code', 'index_name','index_group_no', 'section_no', 'section_name',
    'garment_group_no', 'detail_desc','est_purchase'], axis=1, inplace=True)
    
    not_bought = not_bought.sample(frac=1).reset_index(drop=True)
    
    return not_bought.head(n_recs)

# Second function for content based recommendations
def article_recommend(article_input, n_recs2, return_scores=False, return_explanation=False):
    article = articles_df2[articles_df2['article_id'] == article_input].index
    y = np.array(meta_data.loc[article]).reshape(1, -1)
    
    cos_sim = cosine_similarity(meta_data, y)
    cos_sim = pd.DataFrame(data=cos_sim, index=meta_data.index)
    cos_sim.sort_values(by=0, ascending=False, inplace=True)
    
    # Store scores for hybrid recommendations
    if return_scores:
        scores_df = cos_sim.copy()
        scores_df.columns = ['similarity_score']
    
    results = cos_sim.index.values
    results_df = articles_df2.loc[results]
    results_df.reset_index(inplace=True)
    results_df.rename(columns={'prod_name':'Product Name', 'author':'Author',
                              'product_type_name':'Product Type Name', 'product_group_name':'Product Group Name',
                              'index_group_name':'Index Group Name', 'garment_group_name ':'Garment Group Name'}, inplace=True)
    results_df = results_df.iloc[:100, :]
    
    # Generate explanations for recommended items
    explanations = {}
    if return_explanation:
        # Get details of the input article
        input_article = articles_df2[articles_df2['article_id'] == article_input].iloc[0]
        
        # For each recommended article, create explanation
        for idx, article_id in enumerate(results_df['article_id'].head(n_recs2)):
            rec_item = results_df[results_df['article_id'] == article_id].iloc[0]
            
            # Identify common attributes
            common_attributes = []
            
            if input_article['product_type_name'] == rec_item['Product Type Name']:
                common_attributes.append(f"same product type ({rec_item['Product Type Name']})")
            
            if input_article['product_group_name'] == rec_item['Product Group Name']:
                common_attributes.append(f"same product group ({rec_item['Product Group Name']})")
                
            if input_article['index_group_name'] == rec_item['Index Group Name']:
                common_attributes.append(f"same index group ({rec_item['Index Group Name']})")
            
            if common_attributes:
                explanation = f"This item has the {', '.join(common_attributes)} as article {article_input}."
            else:
                explanation = f"This item has similar characteristics to article {article_input}."
                
            explanations[article_id] = explanation
    
    if return_scores and return_explanation:
        return results_df.head(n_recs2), scores_df, explanations
    elif return_scores:
        return results_df.head(n_recs2), scores_df
    elif return_explanation:
        return results_df.head(n_recs2), explanations
    
    results_df.drop(['product_code', 'product_type_no', 'graphical_appearance_no','graphical_appearance_name', 'colour_group_code', 'colour_group_name',
    'perceived_colour_value_id', 'perceived_colour_value_name','perceived_colour_master_id', 'perceived_colour_master_name',
    'department_no', 'department_name', 'index_code', 'index_name','index_group_no', 'section_no', 'section_name',
    'garment_group_no', 'detail_desc', 'index'], axis=1, inplace=True)
    
    results_df = results_df.sample(frac=1).reset_index(drop=True)
    
    return results_df.head(n_recs2)

# Simplified hybrid recommendation function
def hybrid_recommend(customer, article_input, n_recs, collab_weight=0.6, return_explanation=False):
    """
    Generate hybrid recommendations using both collaborative and content-based filtering
    
    Parameters:
    - customer: customer ID for collaborative filtering
    - article_input: article ID for content-based filtering
    - n_recs: total number of recommendations to return
    - collab_weight: weight for collaborative filtering (0-1)
    - return_explanation: whether to return explanations
    
    Returns:
    - DataFrame with hybrid recommendations
    - Dict of explanations (if return_explanation=True)
    """
    # Calculate how many recommendations should come from each method
    n_collab = math.ceil(n_recs * collab_weight)
    n_content = n_recs - n_collab
    
    collab_explanations = {}
    content_explanations = {}
    
    # Get collaborative filtering recommendations
    try:
        if return_explanation:
            collab_recs, collab_explanations = customer_article_recommend(
                customer, n_collab, return_explanation=True)
        else:
            collab_recs = customer_article_recommend(customer, n_collab)
            
        has_collab = True
    except Exception as e:
        st.warning(f"Could not generate collaborative recommendations: {e}")
        has_collab = False
    
    # Get content-based recommendations
    try:
        if return_explanation:
            content_recs, content_explanations = article_recommend(
                article_input, n_content, return_explanation=True)
        else:
            content_recs = article_recommend(article_input, n_content)
            
        has_content = True
    except Exception as e:
        st.warning(f"Could not generate content-based recommendations: {e}")
        has_content = False
    
    # Combine results based on what's available
    if has_collab and has_content:
        # Combine the two DataFrames
        combined_recs = pd.concat([collab_recs, content_recs])
        
        # Remove duplicates if any (keep first occurrence)
        combined_recs = combined_recs.drop_duplicates(subset=['article_id'])
        
        # If we have more than n_recs after removing duplicates, trim the excess
        if len(combined_recs) > n_recs:
            combined_recs = combined_recs.head(n_recs)
            
        # Shuffle to mix collaborative and content-based recommendations
        combined_recs = combined_recs.sample(frac=1).reset_index(drop=True)
        
        # Combine explanations
        if return_explanation:
            explanations = {**collab_explanations, **content_explanations}
            return combined_recs, explanations
        return combined_recs
    
    # If only one method worked, return those results
    elif has_collab:
        if return_explanation:
            return collab_recs, collab_explanations
        return collab_recs
    elif has_content:
        if return_explanation:
            return content_recs, content_explanations
        return content_recs
    else:
        # If neither worked, return empty DataFrame with proper columns
        empty_df = pd.DataFrame(columns=['article_id', 'Product Name', 'Product Type Name', 
                                    'Product Group Name', 'Index Group Name', 'Garment Group Name'])
        if return_explanation:
            return empty_df, {}
        return empty_df

# Helper function to display recommendations as cards
def display_recommendations(results, n_recs, explanations=None):
    # Remove the tabular display of results since we're using cards now
    # (Keep this line commented out to remove the table)
    # st.table(results)
    
    # Define number of cards per row
    cards_per_row = 3
    
    # Process all recommendations
    for i in range(0, min(len(results), n_recs), cards_per_row):
        # Create columns for this row
        cols = st.columns(cards_per_row)
        
        # Fill each column with a card
        for j in range(cards_per_row):
            idx = i + j
            if idx < min(len(results), n_recs):
                article_id = results.iloc[idx]['article_id']
                
                with cols[j]:
                    # Create card using st.container for styling
                    with st.container():
                        # Add a border and styling with CSS
                        st.markdown("""
                        <style>
                            .card-container {
                                border: 1px solid #e0e0e0;
                                border-radius: 5px;
                                padding: 10px;
                                margin-bottom: 20px;
                                background-color: white;
                                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                            }
                        </style>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("<div class='card-container'>", unsafe_allow_html=True)
                        
                        # Display image 
                        try:
                            img_path = f'Data/h-and-m-personalized-fashion-recommendations/images/0{str(article_id)[:2]}/0{int(article_id)}.jpg'
                            st.image(img_path, use_container_width=True)
                        except Exception:
                            st.image("https://via.placeholder.com/200x250?text=No+Image", use_container_width=True)
                        
                        # Display product information
                        st.markdown(f"### {results.iloc[idx]['Product Name']}")
                        st.markdown(f"**Type:** {results.iloc[idx]['Product Type Name']}")
                        
                        # Additional details can be added here
                        if 'Product Group Name' in results.columns:
                            st.markdown(f"**Group:** {results.iloc[idx]['Product Group Name']}")
                        
                        st.markdown(f"**ID:** `{article_id}`")
                        
                        # Show explanation if available
                        if explanations and article_id in explanations:
                            with st.expander("Recommendation Reason"):
                                st.info(explanations[article_id])
            
                        
                        st.markdown("</div>", unsafe_allow_html=True)

# Create welcome screen and navigation
page_names = ['Home Page', 'Collaborative Filtering', 'Content-Based Filtering', 'Hybrid Recommendations']
page = st.sidebar.radio('Navigation', page_names)

# Replace the existing Home Page section with this enhanced version:

if page == 'Home Page':
    st.title('Welcome to the H&M Recommender System')
    
    st.markdown("""
    ### About This Application
    
    This interactive recommender system helps fashion enthusiasts discover H&M clothing items tailored to their preferences. Using advanced machine learning algorithms, we analyze customer purchase data and product characteristics to provide personalized fashion recommendations.
    
    ### Three Recommendation Approaches
    """)
    
    # Create columns for the three recommendation approaches
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("👥 Collaborative Filtering")
        st.markdown("""
        **How it works**: Analyzes your purchase history and compares it with other customers who have similar preferences.
        
        **Best for**: Discovering new items that similar customers have purchased but you haven't seen yet.
        
        **You'll need**: Your customer ID to access your purchase history.
        """)
    
    with col2:
        st.subheader("👕 Content-Based Filtering")
        st.markdown("""
        **How it works**: Analyzes product attributes like category, color, style, and other features to find similar items.
        
        **Best for**: Finding alternatives to specific products you already like.
        
        **You'll need**: An article ID of an item you're interested in.
        """)
    
    with col3:
        st.subheader("🔄 Hybrid Recommendations")
        st.markdown("""
        **How it works**: Combines both collaborative and content-based approaches for more diverse recommendations.
        
        **Best for**: Getting the most personalized and varied recommendations.
        
        **You'll need**: Both your customer ID and an article ID of interest.
        """)
    
    st.markdown("---")
    
    st.subheader("📊 The Technology Behind Our Recommendations")
    
    st.markdown("""
    #### Collaborative Filtering in Detail
    
    Our collaborative filtering system uses a sophisticated algorithm that:
    
    1. **Analyzes purchase patterns**: We examine which items are frequently bought together or by the same customers.
    2. **Identifies customer similarities**: We group customers with similar tastes based on their purchase history.
    3. **Predicts preferences**: We estimate how likely you are to purchase items you haven't seen based on what similar customers have bought.
    
    This approach is particularly effective at discovering products you might not have found on your own but are likely to enjoy based on your taste profile.
    
    #### Content-Based Filtering in Detail
    
    Our content-based filtering system:
    
    1. **Creates item profiles**: Each H&M article is represented by detailed attributes including product type, style, color, material, and more.
    2. **Measures similarity**: We use cosine similarity to mathematically determine how similar items are to each other.
    3. **Finds alternatives**: When you select an item, we can immediately find the most similar products in the entire H&M catalog.
    
    This approach is excellent for finding variations of items you already like or alternatives when an item is out of stock.
    
    #### Hybrid Recommendations in Detail
    
    Our hybrid system intelligently combines both methods by:
    
    1. **Balancing recommendations**: You can adjust the weight between collaborative and content-based recommendations.
    2. **Removing duplicates**: We ensure you don't see the same recommendation twice.
    3. **Providing explanations**: We tell you why each item was recommended to improve your shopping experience.
    
    This approach provides the most comprehensive set of recommendations, considering both your unique taste profile and specific item preferences.
    """)
    
    st.markdown("---")
    
    st.subheader("📱 How to Use This App")
    
    st.markdown("""
    1. **Navigate** using the sidebar menu to choose your preferred recommendation method.
    2. **Input** your customer ID (for collaborative or hybrid recommendations) or an article ID (for content-based or hybrid recommendations).
    3. **Customize** the number of recommendations you'd like to see.
    4. **Enable explanations** to understand why each item was recommended to you.
    5. **Explore** the recommended items with images and detailed product information.
    
    #### Pro Tips:
    
    - **Try different weights** in the hybrid recommendations to see how it affects your results.
    - **Compare methods** to see which one gives you the most interesting recommendations.
    - **Use the sample IDs** if you don't have your own customer or article IDs.
    """)
    
    st.markdown("---")
    

    st.image("Images/banner2.jpg", caption="Discover your next favorite H&M items with our recommendation system")
    

elif page == 'Collaborative Filtering':
    st.header("Recommendations based on your purchase history")
    
    # Initialize session state for customer input if not already present
    if "customer_input" not in st.session_state:
        st.session_state["customer_input"] = st.session_state.get('customer_id', '')
    
    # Sample customer buttons
    st.subheader("Try one of these sample customers:")
    sample_cols = st.columns(len(SAMPLE_CUSTOMER_IDS))
    
    for i, sample_id in enumerate(SAMPLE_CUSTOMER_IDS):
        with sample_cols[i]:
            if st.button(f"Customer {i+1}", key=f"cust_{i}"):
                st.session_state["customer_input"] = sample_id
                st.session_state['customer_id'] = sample_id  # Keep for persistence
    
    # Use the key that matches session state
    customer_input = st.text_input("Please input your unique Customer ID.", 
                                  key="customer_input")
    
    n_recs = st.number_input("Please enter the number of article recommendations you would like.", 
                            min_value=1, max_value=20, value=5)
    show_explanations = st.checkbox("Show recommendation explanations", value=True)
    
    # Only generate recommendations when the button is clicked
    if st.button("Get Recommendations", key="collab_rec_button"):
        try:
            if customer_input:
                with st.spinner(f'Generating collaborative filtering recommendations...'):
                    if show_explanations:
                        results, explanations = customer_article_recommend(customer_input, n_recs, return_explanation=True)
                        if not results.empty:
                            st.success(f"Successfully generated {len(results)} recommendations!")
                            display_recommendations(results, n_recs, explanations)
                        else:
                            st.warning("No recommendations could be generated. Please try a different customer ID.")
                    else:
                        results = customer_article_recommend(customer_input, n_recs)
                        if not results.empty:
                            st.success(f"Successfully generated {len(results)} recommendations!")
                            display_recommendations(results, n_recs)
                        else:
                            st.warning("No recommendations could be generated. Please try a different customer ID.")
            else:
                st.warning("Please enter a customer ID before generating recommendations.")
        except Exception as e:
            st.error(f"Error generating recommendations: {e}")

elif page == 'Content-Based Filtering':
    st.header("Similar articles to one you like")
    
    # Initialize session state for article input if not already present
    if "article_input" not in st.session_state:
        # Initialize to empty string instead of default value
        st.session_state["article_input"] = ""
    
    # Sample article buttons
    st.subheader("Try one of these sample articles:")
    sample_cols = st.columns(len(SAMPLE_ARTICLE_IDS))
    
    for i, sample_id in enumerate(SAMPLE_ARTICLE_IDS):
        with sample_cols[i]:
            if st.button(f"Article {i+1}", key=f"art_{i}"):
                st.session_state["article_input"] = sample_id
                st.session_state['article_id'] = sample_id  # Keep for persistence
    
    # Use the key that matches session state, but handle empty input
    article_input_str = st.text_input("Please enter an article ID.", 
                                    value=str(st.session_state["article_input"]) if st.session_state["article_input"] else "",
                                    key="article_input_display")
    
    # Convert to integer if not empty
    article_input = int(article_input_str) if article_input_str.strip() else None
    
    # Update session state with manually entered value
    if article_input_str.strip():
        st.session_state["article_input"] = article_input
    
    # Rest of the code remains the same
    n_recs2 = st.number_input("Please enter the number of recommendations you would like.", 
                             min_value=1, max_value=20, value=5, key=2)
    show_explanations = st.checkbox("Show recommendation explanations", value=True, key="cb_explain")
    
    # Only generate recommendations when the button is clicked
    if st.button("Get Recommendations", key="content_rec_button"):
        try:
            if article_input:
                # Display the original article first
                st.subheader("Your selected article:")
                original_article = articles_df2[articles_df2['article_id'] == article_input].iloc[0]
                st.write(f"**Product Name:** {original_article['prod_name']}")
                st.write(f"**Product Type:** {original_article['product_type_name']}")
                
                try:
                    img_path = f'Data/h-and-m-personalized-fashion-recommendations/images/0{str(article_input)[:2]}/0{int(article_input)}.jpg'
                    st.image(img_path, width=200)
                except Exception as e:
                    st.warning(f"Could not display image for selected article: {e}")
                
                st.subheader("Similar recommendations:")
                
                with st.spinner(f'Generating content-based recommendations for similar items...'):
                    if show_explanations:
                        results2, explanations = article_recommend(article_input, n_recs2, return_explanation=True)
                        if not results2.empty:
                            st.success(f"Successfully generated {len(results2)} similar item recommendations!")
                            display_recommendations(results2, n_recs2, explanations)
                        else:
                            st.warning("No similar items could be found. Please try a different article ID.")
                    else:
                        results2 = article_recommend(article_input, n_recs2)
                        if not results2.empty:
                            st.success(f"Successfully generated {len(results2)} similar item recommendations!")
                            display_recommendations(results2, n_recs2)
                        else:
                            st.warning("No similar items could be found. Please try a different article ID.")
            else:
                st.warning("Please enter an article ID before generating recommendations.")
        except Exception as e:
            st.error(f"Error generating recommendations: {e}")

else:  # Hybrid Recommendations
    st.header("Hybrid Recommendations: Best of Both Worlds")
    st.write("This combines recommendations based on your purchase history and articles similar to ones you like.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Initialize session state for hybrid customer input if not already present
        if "hybrid_customer_input" not in st.session_state:
            st.session_state["hybrid_customer_input"] = st.session_state.get('hybrid_customer_id', '')
        
        # Sample customer buttons
        st.subheader("Try a sample customer:")
        sample_cols = st.columns(min(3, len(SAMPLE_CUSTOMER_IDS)))
        
        for i, sample_id in enumerate(SAMPLE_CUSTOMER_IDS[:3]):  # Limit to 3 samples for space
            with sample_cols[i]:
                if st.button(f"Customer {i+1}", key=f"hybrid_cust_{i}"):
                    st.session_state["hybrid_customer_input"] = sample_id
                    st.session_state['hybrid_customer_id'] = sample_id  # Keep for persistence
        
        # Use the key that matches session state
        customer_input = st.text_input("Please input your unique Customer ID.", 
                                      key="hybrid_customer_input")
    
    with col2:
        # Initialize session state for hybrid article input if not already present
        if "hybrid_article_input" not in st.session_state:
            # Initialize to empty string instead of default value
            st.session_state["hybrid_article_input"] = ""
        
        # Sample article buttons
        st.subheader("Try a sample article:")
        sample_cols = st.columns(min(3, len(SAMPLE_ARTICLE_IDS)))
        
        for i, sample_id in enumerate(SAMPLE_ARTICLE_IDS[:3]):  # Limit to 3 samples for space
            with sample_cols[i]:
                if st.button(f"Article {i+1}", key=f"hybrid_art_{i}"):
                    st.session_state["hybrid_article_input"] = sample_id
                    st.session_state['hybrid_article_id'] = sample_id  # Keep for persistence
        
        # Use the key that matches session state, but handle empty input
        article_input_str = st.text_input("Please enter an article ID.", 
                                       value=str(st.session_state["hybrid_article_input"]) if st.session_state["hybrid_article_input"] else "",
                                       key="hybrid_article_input_display")
        
        # Convert to integer if not empty
        article_input = int(article_input_str) if article_input_str.strip() else None
        
        # Update session state with manually entered value
        if article_input_str.strip():
            st.session_state["hybrid_article_input"] = article_input
    
    col3, col4 = st.columns(2)
    
    with col3:
        n_recs = st.number_input("Number of recommendations:", min_value=1, max_value=20, value=5, key="hybrid_recs")
    
    with col4:
        collab_weight = st.slider("Collaborative filtering weight:", 0.0, 1.0, 0.6, 0.1, 
                                 help="Higher values give more importance to your purchase history, lower values focus more on item similarity")
    
    show_explanations = st.checkbox("Show recommendation explanations", value=True, key="hybrid_explain")
    
    # Only generate recommendations when the button is clicked
    if st.button("Get Recommendations", key="hybrid_button"):
        try:
            if customer_input and article_input:
                with st.spinner(f'Generating hybrid recommendations (about {math.ceil(n_recs * collab_weight)} collaborative and {n_recs - math.ceil(n_recs * collab_weight)} content-based)...'):
                    if show_explanations:
                        results, explanations = hybrid_recommend(customer_input, article_input, n_recs, 
                                                        collab_weight, return_explanation=True)
                        if not results.empty:
                            st.success(f"Successfully generated {len(results)} recommendations!")
                            display_recommendations(results, n_recs, explanations)
                        else:
                            st.warning("No recommendations could be generated. Please try different inputs.")
                    else:
                        results = hybrid_recommend(customer_input, article_input, n_recs, collab_weight)
                        if not results.empty:
                            st.success(f"Successfully generated {len(results)} recommendations!")
                            display_recommendations(results, n_recs)
                        else:
                            st.warning("No recommendations could be generated. Please try different inputs.")
            else:
                st.warning("Please enter both a customer ID and an article ID before generating recommendations.")
        except Exception as e:
            st.error(f"Error generating hybrid recommendations: {e}")