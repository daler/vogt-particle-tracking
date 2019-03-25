import pandas as pd
import seaborn
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from scipy.stats import linregress
from scipy import linalg
from matplotlib.collections import LineCollection
from sklearn import mixture


def load_data_from_sampletable(sampletable_filename, uid_columns, **kwargs):
    """
    Assumes that the sampletable has at a column called "filename".

    Loads each filename as a dataframe, with additional kwargs passed to
    pandas.read_table.

    For each loaded dataframe, creates additional columns of any other metadata
    columns for that filename in the sampletable.

    Parameters
    ----------
    sampletable_filename : str

    uid_columns : list
        When these columns are concatenated, they represent a single track.
        "TRACK_ID" will likely be the last item on this list, but the others
        will depend on the experimental design.
    """
    sampletable = pd.read_table(sampletable_filename, **kwargs)
    sampletable.index = sampletable['filename']

    cols_to_attach = [i for i in sampletable.columns if i != 'filename']
    dfs = []
    for filename, row in sampletable.iterrows():
        df = pd.read_table(filename, sep=',')
        df = df.sort_values(['TRACK_ID', 'POSITION_T'])
        for col in cols_to_attach:
            df[col] = row[col]
        dfs.append(df)

    df = pd.concat(dfs, axis=0)
    df = df.reset_index(drop=True)
    df['uid'] = df[uid_columns].apply(lambda x: '_'.join(map(str, x)), axis=1)
    return df


def deltas(data):
    """
    To be applied to a single, unique track and assumed to be sorted by time.

    Adds various calculated values in a per-track manner:

        - d(x, y, z, t): per-timepoint deltas for each coordinate
        - norm(x, y, z, t): subtract starting position from all coordinates to get
          position relative to start
        - normi: mean intensity, after subtracting initial mean intensity
        - perci: mean intensity, transformed into a percentage of the maximum
          total for the track.
        - displacement: per-timepoint total x,y,z displacement
        - speed: displacement / dt
        - normdisplacment: distance from origin

    Parameters
    ----------

    data : pandas.DataFrame
        - Represents a single track.
        - Expected to have POSITION_X, POSITION_Y, POSITION_Z, POSITION_T,
          MEAN_INTENSITY columns.
        - Expected to be sorted by POSITION_T.
    """
    d = pd.DataFrame(
        dict(
            dx=data['POSITION_X'].diff(),
            dy=data['POSITION_Y'].diff(),
            dz=data['POSITION_Z'].diff(),
            dt=data['POSITION_T'].diff(),
            di=data['MEAN_INTENSITY'].diff(),
        )
    )
    t1 = data['POSITION_T'].min()
    imax = data['MEAN_INTENSITY'].max()
    idx = data['POSITION_T'] == t1

    x1 = data.loc[idx, 'POSITION_X'].values[0]
    y1 = data.loc[idx, 'POSITION_Y'].values[0]
    z1 = data.loc[idx, 'POSITION_Z'].values[0]
    i1 = data.loc[idx, 'MEAN_INTENSITY'].values[0]

    d['normx'] = data['POSITION_X'] - x1
    d['normy'] = data['POSITION_Y'] - y1
    d['normz'] = data['POSITION_Z'] - z1
    d['normt'] = data['POSITION_T'] - t1
    d['normi'] = data['MEAN_INTENSITY'] - i1
    d['perci'] = data['MEAN_INTENSITY'] / imax

    d['displacement'] = np.sqrt(d['dx'] ** 2 + d['dy'] ** 2 + d['dz'] ** 2)
    d['speed'] = d['displacement'] / d['dt']

    # distance from starting position
    d['normdisplacement'] = np.sqrt(d['normx'] ** 2 + d['normy'] ** 2 + d['normz'] ** 2)

    return d


def attach_deltas(data):
    """
    Attaches various per-track deltas to the full dataframe.

    Parameters
    ----------
    data : pandas.DataFrame
        Expected to have a "uid" column that is used to uniquely identify
        tracks.

    Returns
    -------

    pandas.DataFrame with additional columns as documented in `deltas`, each
    calculated per track.
    """
    data = data.sort_values(['uid', 'POSITION_T'])
    f = data.groupby('uid', sort=False).apply(deltas)
    data = pd.concat([data, f], axis=1)
    return data.reset_index(drop=True)


def attach_descriptions(data, filename, uid_columns, columns_to_add, **kwargs):
    """
    Given a per-track metadata file, attach optional additional columns to the
    full dataframe.

    Parameters
    ----------
    data : pandas.DataFrame
        Full dataframe

    filename : str
        Filename of metadata

    uid_columns : list
        List of columns in `filename` that, when joined by `_`, creates
        a unique ID.

    columns_to_add : list
        List of columns from the per-track metadata to add to the full dataframe.

    Additional kwargs passed to `pandas.read_table`.

    Returns
    -------

    Full pandas.DataFrame with `columns_to_add` columns.
    """
    interesting = pd.read_table(filename, **kwargs)
    interesting['uid'] = interesting[uid_columns].apply(lambda x: '_'.join(map(str, x)), axis=1)
    interesting = interesting.set_index('uid')
    data = data.join(interesting[columns_to_add], on='uid')
    return data


def params_from_gmm(x):
    """
    Given X and Y values, return the mean and stds for the estimated 2D ellipse
    """
    X = x[['normx', 'normy']].values
    gmm = mixture.GaussianMixture(n_components=1, covariance_type='full')
    gmm.fit(X)
    cov = gmm.covariances_[0]
    mean = gmm.means_[0]
    v, w = linalg.eigh(cov)
    angle = np.arctan2(w[0][1], w[0][0])
    angle = 180. * angle / np.pi
    v = 2. * np.sqrt(2.) * np.sqrt(v)
    return mean, v, angle


def per_track(x):
    """
    Attach various statistics and calculated value to a track.

    Parameters
    ----------

    x : pandas.DataFrame
        Represents a single track. It is expected that the track has had the
        `deltas` function run on it such that it has additional columns like
        `normx`, `normy`, `displacement`, etc.

    Returns
    -------

    pandas.Series with values summarizing the track.
    """
    mn, v, angle = params_from_gmm(x)
    slope, intercept, r_value, p_value, std_err = linregress(x['normt'], x['perci'])
    s = pd.Series(
        dict(
            slope=slope,
            intercept=intercept,
            r_value=r_value,
            p_value=p_value,
            std_err=std_err,
            max_normy=x['normy'].max(),
            duration=len(x),
            start=x['POSITION_T'].min(),
            end=x['POSITION_T'].max(),
            total_disp=x['displacement'].sum(),
            total_abs_disp=x['displacement'].abs().sum(),
            max_disp=x['displacement'].max(),
            max_normdisp=x['normdisplacement'].max(),
            min_normdisp=x['normdisplacement'].min(),
            avg_normdisp=x['normdisplacement'].mean(),
            centerx=mn[0],
            centery=mn[1],
            v0=v[0],
            v1=v[1],
            maxv=max(v),
            minv=min(v),
            eccentricity=max(v) / min(v),
            angle=angle,
        )
    )
    s['rsquared'] = s['r_value'] ** 2
    s['avg_abs_disp'] = s['total_abs_disp'] / s['duration']
    s['avg_disp'] = s['total_disp'] / s['duration']
    return s


def attach_per_track(data, uid_columns):
    m = data.groupby('uid').apply(per_track)
    data_full = data.join(m, on='uid')
    by_uid = m.join(data[uid_columns + ['TRACK_ID', 'uid']].set_index('uid')).drop_duplicates()
    return data_full, by_uid


def myscatter(x, y, data, colorby=None, cmap='viridis', norm=None,
              plot_ellipse=False, scatterkws=dict(), collkws=dict(),
              ellipsekws=dict(), **kws):
    """
    Plotting function to be applied to each facet of a seaborn.FacetGrid.

    Parameters
    ----------
    x, y : str
        Columns of dataframe to plot
    data : pandas.DataFrame
        Subset of full dataframe passed by FacetGrid.map_dataframe
    colorby : None or str
        If not None, the line segments will be colored according to this column
    cmap : str
        matplotlib colormap to use when plotting segments
    norm : matplotlib normalize object
        Only applies if `colorby` is not None. If `norm` is None, then each
        line's values will be normalized from min to max. Otherwise, the
        norm's vmin and vmax will be used to force the same colormap across
        all facets.
    kws : dict
        Additional keyword args are passed to plt.scatter
    """
    ax = plt.gca()
    for uid, group in data.groupby('uid'):
        xy = np.column_stack([group[x], group[y]]).reshape(-1, 1, 2)
        segments = np.hstack([xy[:-1], xy[1:]])
        col = LineCollection(segments, cmap=cmap, norm=norm, **collkws)
        if colorby is not None:
            col.set_array(group[colorby])
        ax.add_collection(col)
        _kws = dict(c='.5', s=2)
        _kws.update(**scatterkws)
        facetgrid_color = _kws.pop('color', None)
        ax.scatter(group[x], group[y], **_kws)
        if plot_ellipse:
            mean = group[['centerx', 'centery']].drop_duplicates().values
            v = group[['v0', 'v1']].drop_duplicates().values
            angle = group[['angle']].drop_duplicates().values[0]
            ell = matplotlib.patches.Ellipse(mean[0], v[0][0], v[0][1], 180.+angle, **ellipsekws)
            plt.gca().add_artist(ell)
    xmax = data[x].max()
    ax.axis('tight')


def facetplotter(x, y, data, colorby=None, norm=None, scatterkws=dict(),
                 collkws=dict(), ellipsekws=dict(), vmin=None, vmax=None,
                 plot_ellipse=False, **kws):
    g = seaborn.FacetGrid(data, **kws)
    if colorby and norm is not None:
        norm = plt.Normalize(vmin=data[colorby].min(), vmax=data[colorby].max())
        if vmin is not None:
            norm.vmin = vmin
        if vmax is not None:
            norm.vmax = vmax
    g = g.map_dataframe(myscatter, x, y, colorby=colorby, norm=norm,
                        plot_ellipse=plot_ellipse, scatterkws=scatterkws,
                        collkws=collkws, ellipsekws=ellipsekws)


def points(x, y, data, colorby=None, cmap='viridis', norm=None, **kws):
    """
    Plotting function to be applied to each facet of a seaborn.FacetGrid.

    Parameters
    ----------
    x, y : str
        Columns of dataframe to plot
    data : pandas.DataFrame
        Subset of full dataframe passed by FacetGrid.map_dataframe
    colorby : None or str
        If not None, the line segments will be colored according to this column
    cmap : str
        matplotlib colormap to use when plotting segments
    norm : matplotlib normalize object
        Only applies if `colorby` is not None. If `norm` is None, then each
        line's values will be normalized from min to max. Otherwise, the
        norm's vmin and vmax will be used to force the same colormap across
        all facets.
    kws : dict
        Additional keyword args are passed to plt.scatter
    """
    ax = plt.gca()
    x = data[x]
    y = data[y]
    bins = kws.get('bins', None)
    if bins == 'log':
        label = 'log10(density)'
    else:
        label = 'density'
    mappable = ax.hexbin(x, y, **kws)
    plt.colorbar(mappable, label=label)
    ax.axis('tight')


def pointplotter(x, y, data, colorby=None, norm=None, scatterkws=dict(), vmin=None, vmax=None, **kws):
    g = seaborn.FacetGrid(data, **kws)
    if colorby and norm is not None:
        norm = plt.Normalize(vmin=data[colorby].min(), vmax=data[colorby].max())
        if vmin is not None:
            norm.vmin = vmin
        if vmax is not None:
            norm.vmax = vmax
    g = g.map_dataframe(points, x, y, colorby=colorby, norm=norm, **scatterkws)
    return g
