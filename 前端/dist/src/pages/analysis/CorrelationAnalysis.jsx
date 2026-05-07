import { useEffect, useRef, useState, useCallback, useMemo } from 'react'
import { Card, Select, DatePicker, Button, Row, Col, Spin, message, Alert } from 'antd'
import ReactECharts from 'echarts-for-react'
import dayjs from 'dayjs'
import { analyticsApi, getApiErrorMessage } from '../../api'

const HEATMAP_GRADIENT = ['#313695', '#4575b4', '#74add1', '#abd9e9', '#e0f3f8', '#fee090', '#fdae61', '#f46d43', '#d73027']

function matrixToHeatmapData(mat) {
  const data = []
  const n = mat.length
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) data.push([j, i, mat[i][j]])
  return data
}

export default function CorrelationAnalysis() {
  const [loading, setLoading] = useState(true)
  const [loadingScatter, setLoadingScatter] = useState(false)
  const [dayType, setDayType] = useState('all')
  const [labels, setLabels] = useState([])
  const [matrix, setMatrix] = useState([])
  const [scatter, setScatter] = useState([])
  const [scatterErr, setScatterErr] = useState('')
  const [scatterStart, setScatterStart] = useState(null)
  /** 散点预测步数：越大点越多（后端最大 192） */
  const [scatterSteps, setScatterSteps] = useState(96)
  const [locationLabel, setLocationLabel] = useState('')
  const heatmapRef = useRef(null)
  const scatterRef = useRef(null)
  const scatterStartRef = useRef(scatterStart)
  scatterStartRef.current = scatterStart

  const loadHeatmap = useCallback(async () => {
    setLoading(true)
    try {
      const loc = await analyticsApi.location()
      setLocationLabel(loc.data?.label || '')
      const cr = await analyticsApi.correlation(dayType)
      setLabels(cr.data.labels || [])
      setMatrix(cr.data.matrix || [])
    } catch (e) {
      message.error(getApiErrorMessage(e))
    } finally {
      setLoading(false)
    }
  }, [dayType])

  const loadScatter = useCallback(async () => {
    setScatterErr('')
    const ss = scatterStartRef.current
    const start = ss ? ss.startOf('day').toISOString() : dayjs().startOf('day').toISOString()
    setLoadingScatter(true)
    setScatter([])
    try {
      const sc = await analyticsApi.modelScatter(start, scatterSteps)
      const pts = sc.data.points || []
      setScatter(pts)
      if (!pts.length) setScatterErr('散点数据为空：请检查后端预测服务与模型文件。')
    } catch (e) {
      const msg = getApiErrorMessage(e)
      setScatterErr(msg)
      message.warning(`散点图：${msg}`)
    } finally {
      setLoadingScatter(false)
    }
  }, [scatterSteps])

  useEffect(() => {
    void loadHeatmap()
  }, [loadHeatmap])

  useEffect(() => {
    void loadScatter()
  }, [loadScatter])

  const handleQuery = () => {
    void loadHeatmap()
    void loadScatter()
  }

  const heatmapOption = {
    backgroundColor: 'transparent',
    title: { text: `负荷-气象相关性（${locationLabel || '训练集'}）`, left: 'center', textStyle: { color: '#e8f4ff' } },
    tooltip: { backgroundColor: 'rgba(10,22,40,0.9)' },
    grid: { left: '18%', right: '12%', top: '15%', bottom: '18%' },
    xAxis: { type: 'category', data: labels, axisLabel: { color: '#e8f4ff', rotate: 30 } },
    yAxis: { type: 'category', data: labels, axisLabel: { color: '#e8f4ff' } },
    visualMap: { min: -1, max: 1, inRange: { color: HEATMAP_GRADIENT }, textStyle: { color: '#e8f4ff' } },
    series: [{ type: 'heatmap', data: matrixToHeatmapData(matrix), emphasis: { itemStyle: { borderColor: '#ff4d4f', borderWidth: 2 } } }],
  }

  /** 后端 points 为 [[main_mw, baseline_mw], ...]，每点已是 (横=主模型, 纵=持久化)；勿拆成两半当两个模型 */
  const scatterOption = useMemo(() => {
    const pts = Array.isArray(scatter) ? scatter : []
    const flat = pts.flatMap((p) => (Array.isArray(p) ? p : []))
    const lo = flat.length ? Math.min(...flat) : 0
    const hi = flat.length ? Math.max(...flat) : 100
    const pad = Math.max((hi - lo) * 0.05, 1)
    const axisMin = Math.max(0, lo - pad)
    const axisMax = hi + pad
    return {
      backgroundColor: 'transparent',
      title: {
        text: `主模型（TCN-BiLSTM-Attention）vs 持久化基线（散点 · ${scatterSteps} 个 15min 点）`,
        subtext:
          pts.length > 0
            ? `共 ${pts.length} 个点；贴近虚线表示主模型与持久化基线接近（多点重合时颜色会加深）`
            : '',
        left: 'center',
        top: 4,
        textStyle: { color: '#e8f4ff', fontSize: 14 },
        subtextStyle: { color: 'rgba(232,244,255,0.55)', fontSize: 11, lineHeight: 16 },
      },
      tooltip: {
        trigger: 'item',
        backgroundColor: 'rgba(10,22,40,0.92)',
        formatter: (p) => {
          if (!p.data || !Array.isArray(p.data)) return ''
          const [mx, py] = p.data
          const i = p.dataIndex
          return `第 ${i + 1} 步<br/>主模型: ${mx} MW<br/>持久化: ${py} MW<br/>差值 (主−持久): ${(Number(mx) - Number(py)).toFixed(2)} MW`
        },
      },
      legend: {
        data: ['预测点 (横=主模型, 纵=持久化)', 'y = x（完全一致）'],
        textStyle: { color: '#e8f4ff' },
        top: 52,
      },
      grid: {
        containLabel: true,
        left: 16,
        right: 20,
        top: 108,
        bottom: 48,
      },
      xAxis: {
        type: 'value',
        name: '主模型 (MW)',
        nameLocation: 'middle',
        nameGap: 36,
        min: axisMin,
        max: axisMax,
        axisLabel: { color: '#e8f4ff' },
        splitLine: { lineStyle: { color: 'rgba(255,255,255,0.1)' } },
        nameTextStyle: { color: '#e8f4ff', fontSize: 12 },
      },
      yAxis: {
        type: 'value',
        name: '持久化基线 (MW)',
        nameLocation: 'middle',
        nameGap: 44,
        min: axisMin,
        max: axisMax,
        axisLabel: { color: '#e8f4ff' },
        splitLine: { lineStyle: { color: 'rgba(255,255,255,0.1)' } },
        nameTextStyle: { color: '#e8f4ff', fontSize: 12 },
      },
      series: [
        {
          name: '预测点 (横=主模型, 纵=持久化)',
          type: 'scatter',
          data: pts,
          itemStyle: {
            color: 'rgba(64, 169, 255, 0.55)',
            borderColor: 'rgba(145, 213, 255, 0.35)',
            borderWidth: 0.5,
          },
          symbolSize: 8,
          emphasis: {
            itemStyle: { color: '#69c0ff', opacity: 1, borderWidth: 1 },
            scale: 1.35,
          },
        },
        {
          name: 'y = x（完全一致）',
          type: 'line',
          data: [
            [axisMin, axisMin],
            [axisMax, axisMax],
          ],
          lineStyle: { type: 'dashed', color: 'rgba(255,255,255,0.45)', width: 1.5 },
          symbol: 'none',
          tooltip: { show: false },
          emphasis: { disabled: true },
        },
      ],
    }
  }, [scatter, scatterSteps])

  const exportPNG = (which) => {
    const ref = which === 'heatmap' ? heatmapRef : scatterRef
    if (ref.current?.getEchartsInstance()) {
      const url = ref.current.getEchartsInstance().getDataURL({ type: 'png', pixelRatio: 2 })
      const a = document.createElement('a')
      a.href = url
      a.download = `${which === 'heatmap' ? '相关性热力图' : '散点图'}.png`
      a.click()
      message.success('已导出 PNG')
    } else message.warning('图表未加载')
  }

  const exportExcel = () => {
    if (!labels.length || !matrix.length) return
    const csv = [labels.join(','), ...matrix.map((row) => row.join(','))].join('\n')
    const blob = new Blob(['\ufeff' + csv], { type: 'text/csv;charset=utf-8' })
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = '相关性矩阵.csv'
    a.click()
    URL.revokeObjectURL(a.href)
    message.success('已导出 CSV')
  }

  return (
    <div>
      <Card className="glass-card" style={{ marginBottom: 24 }}>
        <Row gutter={16} align="middle">
          <Col><span style={{ color: '#e8f4ff' }}>区域：</span><span style={{ color: '#91d5ff' }}>{locationLabel || '—'}</span></Col>
          <Col><span style={{ color: '#e8f4ff' }}>日类型：</span><Select style={{ width: 120 }} value={dayType} onChange={setDayType} options={[{ value: 'all', label: '全部' }, { value: 'normal', label: '正常日' }, { value: 'weekend', label: '周六日' }, { value: 'holiday', label: '节假日' }]} /></Col>
          <Col><span style={{ color: '#e8f4ff' }}>散点预测起始：</span><DatePicker value={scatterStart} onChange={setScatterStart} allowClear /></Col>
          <Col>
            <span style={{ color: '#e8f4ff' }}>散点速度：</span>
            <Select
              style={{ width: 200 }}
              value={scatterSteps}
              onChange={setScatterSteps}
              options={[
                { value: 48, label: '较快（48 步，约半天）' },
                { value: 96, label: '默认（96 步，全天）' },
                { value: 192, label: '最密（192 步，两天）' },
              ]}
            />
          </Col>
          <Col><Button type="primary" className="tech-btn" onClick={handleQuery}>查询</Button></Col>
          <Col><Button onClick={() => exportPNG('heatmap')}>导出热力图 PNG</Button></Col>
          <Col><Button onClick={() => exportPNG('scatter')}>导出散点 PNG</Button></Col>
          <Col><Button onClick={exportExcel}>导出矩阵 CSV</Button></Col>
        </Row>
        <div style={{ color: 'rgba(255,255,255,0.5)', fontSize: 12, marginTop: 12 }}>
          提示：更换「散点预测起始」后请点击「查询」。散点步数越大图上点越多；主模型与持久化接近时点会沿 y=x 重合，可改用「最密（192 步）」或观察悬停 tooltip。
        </div>
      </Card>
      <Spin spinning={loading}>
        <Row gutter={24}>
          <Col span={12}><Card className="glass-card"><ReactECharts ref={heatmapRef} option={heatmapOption} style={{ height: 320 }} notMerge /></Card></Col>
          <Col span={12}>
            <Card className="glass-card">
              <Spin spinning={loadingScatter}>
                {scatterErr && (
                  <Alert type="warning" showIcon message={scatterErr} style={{ marginBottom: 12 }} />
                )}
                <ReactECharts ref={scatterRef} option={scatterOption} style={{ height: 400 }} notMerge />
              </Spin>
            </Card>
          </Col>
        </Row>
      </Spin>
    </div>
  )
}
