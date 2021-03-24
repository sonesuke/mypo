from mypo import Loader
import pymc3 as pm


loader = Loader()
loader.get(ticker="VOO", expense_ratio=0.0003)
loader.get(ticker="IEF", expense_ratio=0.0015)

market = loader.get_market()
up_rate = market.get_rate_of_change()
up_rate = up_rate[:200]

with pm.Model() as model:
    # LKJCholeskyCovは下三角行列の要素をリストで返す
    packed_L = pm.LKJCholeskyCov("packed_L", n=2, eta=1., sd_dist=pm.HalfCauchy.dist(beta=2.5))
    # 下三角行列に変換する
    L = pm.expand_packed_triangular(2, packed_L, lower=True)
    # 共分散行列にする
    sigma = pm.Deterministic('sigma', L.dot(L.T))

    prior_mu = pm.Uniform("prior_mu", -1, 1, shape=2)
    mu= pm.Normal("returns", mu=prior_mu, sd=1, shape=2)

    obs = pm.MvNormal("observed_returns", mu=mu, chol=L, observed=up_rate)
    step = pm.NUTS()
    trace = pm.sample(100, step)

pm.traceplot(trace)