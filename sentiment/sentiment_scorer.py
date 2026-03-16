import re
from typing import Dict, Tuple

# Simple finance-aware sentiment lexicon
POSITIVE_WORDS = {
    'gain', 'profit', 'surge', 'jump', 'beat', 'strong', 'bullish', 'upgrade',
    'growth', 'momentum', 'rally', 'recovery', 'outperform', 'excel', 'record',
    'win', 'success', 'approve', 'acquire', 'merge', 'boost', 'improve'
}

NEGATIVE_WORDS = {
    'loss', 'decline', 'drop', 'miss', 'weak', 'bearish', 'downgrade', 'crash',
    'recession', 'selloff', 'bankruptcy', 'fail', 'fraud', 'scandal', 'lawsuit',
    'loss', 'slump', 'plunge', 'warning', 'concern', 'risk', 'writedown', 'cut'
}

MODIFIERS = {
    'significantly': 1.5,
    'dramatically': 1.5,
    'strongly': 1.3,
    'slightly': 0.7,
    'modestly': 0.7
}

class SentimentScorer:
    """
    Simple finance-aware sentiment scorer based on keyword matching.
    Scores articles on polarity (-1 to 1), strength, and relevance.
    """
    
    def __init__(self):
        self.positive_words = POSITIVE_WORDS
        self.negative_words = NEGATIVE_WORDS
        self.modifiers = MODIFIERS
    
    def score(self, text: str, headline: str = None) -> Dict:
        """
        Score sentiment on article text and headline.
        Returns: {
            'polarity': -1 to 1,
            'strength': 0 to 1 (confidence),
            'relevance': 0 to 1,
            'novelty_score': 0 to 1
        }
        """
        # Combine text and headline for scoring
        full_text = (headline or '') + ' ' + text
        full_text_lower = full_text.lower()
        
        # Count positive and negative words
        pos_count = self._count_words(full_text_lower, self.positive_words)
        neg_count = self._count_words(full_text_lower, self.negative_words)
        
        # Apply modifiers for emphasis
        pos_score = self._apply_modifiers(full_text_lower, pos_count)
        neg_score = self._apply_modifiers(full_text_lower, neg_count)
        
        # Calculate polarity (-1 to 1)
        total = pos_score + neg_score
        if total == 0:
            polarity = 0.0
        else:
            polarity = (pos_score - neg_score) / total
        
        # Calculate strength (how confident the signal is)
        strength = min(1.0, total / 20.0)  # Normalize to 0-1
        
        # Calculate relevance (headline importance)
        relevance = self._calculate_relevance(headline or '')
        
        # Calculate novelty (not yet widely known)
        novelty = self._calculate_novelty(full_text_lower)
        
        return {
            'polarity': polarity,
            'strength': strength,
            'relevance': relevance,
            'novelty': novelty,
            'raw_score': polarity * strength
        }
    
    def _count_words(self, text: str, word_set: set) -> int:
        """Count occurrences of words from word_set in text."""
        count = 0
        for word in word_set:
            # Use word boundaries
            pattern = r'\b' + word + r'\b'
            count += len(re.findall(pattern, text))
        return count
    
    def _apply_modifiers(self, text: str, base_count: int) -> float:
        """Apply strength modifiers to word count."""
        score = float(base_count)
        for modifier, multiplier in self.modifiers.items():
            if modifier in text:
                score *= multiplier
        return score
    
    def _calculate_relevance(self, headline: str) -> float:
        """
        Relevance based on headline keywords indicating importance.
        High-impact events get higher scores.
        """
        high_impact_keywords = {
            'earnings', 'guidance', 'acquisition', 'merger', 'ipo', 'bankruptcy',
            'fraud', 'lawsuit', 'ceo', 'management', 'regulatory', 'warning',
            'dividend', 'split', 'recall', 'deal', 'breakthrough'
        }
        
        headline_lower = headline.lower()
        impact_count = sum(1 for kw in high_impact_keywords if kw in headline_lower)
        return min(1.0, impact_count * 0.2)
    
    def _calculate_novelty(self, text: str) -> float:
        """
        Novelty score based on typical keywords indicating fresh news.
        """
        novel_keywords = ['breaking', 'just', 'today', 'announced', 'revealed', 'released']
        novelty_count = sum(1 for kw in novel_keywords if kw in text)
        return min(1.0, novelty_count * 0.3)
    
    def weighted_score(self, sentiment_data: Dict) -> float:
        """
        Calculate final weighted sentiment score.
        Combines polarity, strength, relevance, and novelty.
        """
        score = (
            sentiment_data['polarity'] * 0.5 +
            sentiment_data['strength'] * 0.2 +
            sentiment_data['relevance'] * 0.2 +
            sentiment_data['novelty'] * 0.1
        )
        return score
