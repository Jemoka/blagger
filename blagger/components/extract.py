from ..inference.p_rank import P_RANK
from ..inference.qa import QA
from .pdf import extract_fig_mention, clean_label

def select_by_nl(query, sents, figures=[], 
                 threshold=90, topn=5,
                 fallback_threshold=0.1):
    """Natural language relavance selection

    Parameters
    ----------
    query : str
        The text query to search on.
    sents : list
        A list of text
    figures : list, optional
        A list of figures
    threshold : float, optional
        The threshold to return a result.
    fallback_threshold : float, optional
        The threshold to return a result.
    topn : int, optional
        The top n of text identification keep.

    Results
    -------
    List[dict], optional
        If the result crosses the threshold, return the relavent figure(s).
    """

    # extract figure ids and mentions
    fig_ids = [extract_fig_mention(i["caption"]) for i in figures]

    # extract best text scores
    text_scores = P_RANK(documents=sents, question=query)
    best_text_scores = sorted(filter(lambda x:x["score"] > threshold, text_scores),
                            key=lambda x:x["score"], reverse=True)[:topn]
    # if no good performance were returned, fallback to slower QA task
    if len(best_text_scores) == 0:
        print("No answers found, falling back to slower QA process...")
        answers = []
        # we have to run QA per element
        for i in sents:
            answers.append(QA(context=i,
                              question=query))

        answers = sorted(enumerate(answers), key=lambda x:x[1]["score"], reverse=True)
        answer_ids = [i[0] for i in answers[:topn] if i[1]["score"] >= fallback_threshold]
        best_text_scores = [{"document": sents[i]} for i in answer_ids]
    fig_mentions = [i for i in
                    [extract_fig_mention(i["document"]) for i in best_text_scores] if i]
    best_text = [i["document"] for i in best_text_scores]

    # get captions and text scores for caption
    if len(figures) >= 1:
        captions = [clean_label(i["caption"]) for i in figures]
        fig_scores = P_RANK(documents=captions, question=query)
        best_fig_scores = sorted(filter(lambda x:x[1]["score"] > threshold, enumerate(fig_scores)),
                                key=lambda x:x[1]["score"], reverse=True)[:topn]
        fig_rels = [fig_ids[i[0]] for i in best_fig_scores]

        # combine final relavent figures
        rel_fig_indicies = list(set(fig_mentions+fig_rels))
        # and get actual index
        fig_indicies = [fig_ids.index(i) for i in rel_fig_indicies]

        figs = [figures[i] for i in fig_indicies]
    else:
        figs = []

    return best_text, figs

