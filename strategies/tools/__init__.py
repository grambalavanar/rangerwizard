"""
strategies/tools — unified indicator library

Re-exports everything from momentum_tools and mean_reversion_tools and
adds the new regime_tools module.
"""

from momentum.momentum_tools import (
    ema, sma, wma,
    rsi, stochastic_oscillator, stochrsi,
    williams_r, roc, awesome_oscillator, tsi,
    cci, ultimate_oscillator, ppo, kama,
    macd, adx, aroon, parabolic_sar, kst,
    sharpe_ratio, sortino_ratio, max_drawdown,
    calmar_ratio, cagr, win_rate, kelly_criterion,
    volatility, value_at_risk, conditional_value_at_risk,
    momentum_score, rank_momentum, dual_momentum,
    crossover, crossunder, atr, zscore,
)

from mean_reversion.mean_reversion_tools import (
    price_zscore, bollinger_bands, efficiency_ratio,
    hurst_exponent, ou_halflife, variance_ratio,
    connors_rsi, rsi2, volume_climax,
    rsi_bullish_divergence, macd_bullish_divergence,
    mean_reversion_regime,
    avg_holding_period, mean_reversion_speed,
)

from strategies.tools.regime_tools import (
    GaussianHMM, RegimeClassifier,
    volatility_regime, trend_regime, classify_regime,
    REGIME_MOMENTUM, REGIME_MEAN_REVERSION, REGIME_CASH,
)
