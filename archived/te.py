

def create_per_query_dataset(df_dataset_per_query, target_query_id, target_query, validator_document_id, validator_document, target_document_rank, model, boosting_sentences, cross_reranker_sample_1000_query, collection, sent_position):

    # Create dataset_per_query (different position, repalce content, rerank and append to the per query dataset)

    for sent in boosting_sentences: #different new sent

        new_validator_document = sent_position_function(sent, sent_position, validator_document) # new content is created with placing sent in sent_position

        target_query_cross_reranker = cross_reranker_sample_1000_query[cross_reranker_sample_1000_query['query_id'] == target_query_id] #1000 docs for a query
        target_query_cross_reranker['new_new_distance'] = target_query_cross_reranker['distance']

        target_query_cross_reranker_modified = target_doc_content_replacement(target_query_cross_reranker, target_query_id, validator_document_id, collection, new_validator_document) #replace content
        
        target_document_for_new_rerank = target_query_cross_reranker_modified[target_query_cross_reranker_modified['doc_id'] == validator_document_id] #only target document
        target_reranked_score = reranker_score(model, target_document_for_new_rerank, target_query) #get the score of modified doc from cross encoder 
        

        # Replace score
        target_query_cross_reranker.loc[target_query_cross_reranker['doc_id'] == validator_document_id, 'new_new_distance'] = target_reranked_score
        
        # Rank the rows based on 'new_distance' column (descending order)
        target_query_cross_reranker['new_rank'] = target_query_cross_reranker['new_distance'].rank(ascending=False).astype(int)

        
        df_dataset_per_query = append_to_df_dataset_per_query(target_query_cross_reranker, target_query, target_document_rank, collection, sent, sent_position, new_validator_document, df_dataset_per_query) #append to the dataset for this query
        

    return df_dataset_per_query



def target_doc_content_replacement(target_query_cross_reranker, target_query_id, validator_document_id, collection, new_validator_document):
    
    #cross_reranker for that query_id only
    
    target_query_cross_reranker = target_query_cross_reranker.merge(collection, on="doc_id", how="inner")
    target_query_cross_reranker = target_query_cross_reranker[['query_id', 'query', 'doc_id', 'doc_content', 'rank', 'distance']]

    # Replace content
    target_query_cross_reranker.loc[target_query_cross_reranker['doc_id'] == validator_document_id, 'doc_content'] = new_validator_document

    return target_query_cross_reranker #1000 docs for a query with modified doc content


def reranker_score(model, target_document_for_rerank, target_query):
    # Extract the single row
    single_row = target_document_for_rerank.iloc[0]
    
    # Prepare input for the model
    query = target_query
    doc_content = single_row['doc_content']
    list_of_docs = [(query, doc_content)]
    
    # Get the score from the model
    score = model.predict(list_of_docs).tolist()[0]
    
    return score





def append_to_df_dataset_per_query(target_reranked_run_df, target_query, target_document_rank, collection, sent, sent_position, new_validator_document, df_dataset_per_query):

    full_target_reranked_run_df = target_reranked_run_df
    full_target_reranked_run_df['query'] = target_query

    full_target_reranked_run_df = full_target_reranked_run_df.merge(collection, on="doc_id", how="inner")
    full_target_reranked_run_df = full_target_reranked_run_df[['query_id', 'query', 'doc_id','doc_content', 'rank','distance','new_rank', 'new_distance']]
    #full_target_reranked_run_df.columns

    selected_row =  full_target_reranked_run_df[full_target_reranked_run_df['rank'] == target_document_rank]
    selected_row = selected_row.copy()
    selected_row['new_sent'] = sent
    selected_row['new_sent_position'] = sent_position
    selected_row['new_doc_content'] = new_validator_document

    selected_row = selected_row[['query_id', 'query', 'doc_id','doc_content', 'rank','distance', 'new_sent' , 'new_sent_position' ,'new_doc_content', 'new_rank', 'new_distance']]
    #print(selected_row)

    #selected_row['doc_content'].values[0]
    #selected_row['new_doc_content'].values[0]

    df_dataset_per_query = pd.concat([df_dataset_per_query, selected_row], ignore_index=True)
    return df_dataset_per_query

