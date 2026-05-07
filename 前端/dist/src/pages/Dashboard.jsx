import { useRef, useState, useEffect, useMemo } from 'react'
import { Card, Table, Tabs, Spin, message, DatePicker, Button, Alert, Empty, Select } from 'antd'
import AlertModal from '../components/AlertModal'
import ReactECharts from 'echarts-for-react'
import { predictionApi, getApiErrorMessage } from '../api'
import { activeCompareKeys } from '../predictionModels'
import dayjs from 'dayjs'

export default function Dashboard() {
  const [loading, setLoading] = useState(true)
  const [data, setData] = useState([])
  const [weather, setWeather] = useState(null)
  const [loadHint, setLoadHint] = useState('')
  const [compareModelsLoaded, setCompareModelsLoaded] = useState(null)
  const [startDate, setStartDate] = useState(dayjs())
  const [predSteps, setPredSteps] = useState(96)

  useEffect(() => { loadData() }, [startDate, predSteps])

  const loadData = async () => {
    setLoading(true)
    setLoadHint('')
    try {
      const start = startDate.startOf('day').toISOString()
      const res = await predictionApi.getTwoDays(start, predSteps)
      const body = res?.data
      const rows = body?.data
      setCompareModelsLoaded(body?.compare_models_loaded ?? null)
      if (!Array.isArray(rows)) {
        setData([])
        setWeather(body?.weather ?? null)
        setLoadHint(
          '接口返回数据异常（可能未走 Vite 代理、后端未启动或返回了非 JSON）。请用 npm run dev 打开 http://localhost:5173，并确认后端在 8000 端口运行。',
        )
        return
      }
      setData(rows)
      setWeather(body?.weather ?? null)
      if (rows.length === 0) {
        setLoadHint('预测结果为空：请查看后端终端是否有报错（模型路径、依赖等）。')
      }
    } catch (e) {
      const tip = getApiErrorMessage(e)
      message.error(tip)
      setData([])
      setWeather(null)
      setCompareModelsLoaded(null)
      setLoadHint(tip)
    } finally {
      setLoading(false)
    }
  }

  const compareKeysOnData = useMemo(() => activeCompareKeys(data[0]), [data])

  const chartOption = useMemo(() => {
    const series = [
      {
        name: 'TCN-BiLSTM-Attention',
        type: 'line',
        data: data.map((d) => d.main_mw),
        smooth: true,
        lineStyle: { color: '#40a9ff', width: 3 },
        itemStyle: { color: '#40a9ff' },
        z: 10,
        areaStyle: { color: 'rgba(64,169,255,0.12)' },
      },
    ]
    compareKeysOnData.forEach(({ key, short, color }) => {
      series.push({
        name: short,
        type: 'line',
        data: data.map((d) => d[key]),
        smooth: true,
        lineStyle: { color, width: 1.5, type: 'solid' },
        itemStyle: { color },
        symbol: 'none',
      })
    })
    series.push({
      name: '持久化模型',
      type: 'line',
      data: data.map((d) => d.baseline_mw),
      smooth: true,
      lineStyle: { color: '#ff4d4f', width: 1.5, type: 'dashed' },
      itemStyle: { color: '#ff4d4f' },
      symbol: 'none',
    })
    const legend = ['TCN-BiLSTM-Attention', ...compareKeysOnData.map((c) => c.short), '持久化模型']
    return {
      backgroundColor: 'transparent',
      title: {
        text: `多模型负荷预测（${predSteps} 个 15min 点${predSteps <= 96 ? ' · 快速预览' : ' · 完整两天'}）`,
        left: 'center',
        textStyle: { color: '#e8f4ff' },
      },
      tooltip: { trigger: 'axis', backgroundColor: 'rgba(10,22,40,0.9)' },
      legend: {
        data: legend,
        top: 28,
        type: 'scroll',
        textStyle: { color: '#e8f4ff' },
      },
      grid: { left: '3%', right: '4%', bottom: '3%', containLabel: true },
      xAxis: {
        type: 'category',
        data: data.map((d) => d.datetime),
        boundaryGap: false,
        axisLine: { lineStyle: { color: 'rgba(255,255,255,0.3)' } },
        axisLabel: { color: '#e8f4ff', rotate: data.length > 48 ? 45 : 0 },
      },
      yAxis: {
        type: 'value',
        name: '负荷 (MW)',
        axisLine: { show: false },
        splitLine: { lineStyle: { color: 'rgba(255,255,255,0.1)' } },
        axisLabel: { color: '#e8f4ff' },
        nameTextStyle: { color: '#e8f4ff' },
      },
      series,
      animation: true,
    }
  }, [data, predSteps, compareKeysOnData])

  const day1 = data.filter((_, i) => i < 96)
  const day2 = data.filter((_, i) => i >= 96)

  const tableColumns = useMemo(() => {
    const cols = [
      { title: '时间', dataIndex: 'datetime', width: 158, fixed: 'left' },
      { title: '本文主模型 (MW)', dataIndex: 'main_mw', width: 148 },
    ]
    compareKeysOnData.forEach(({ key, short }) => {
      cols.push({ title: `${short} (MW)`, dataIndex: key, width: 112 })
    })
    cols.push(
      { title: '持久化 (MW)', dataIndex: 'baseline_mw', width: 108 },
      {
        title: '主−持久 (MW)',
        key: 'diff',
        width: 108,
        render: (_, r) => (r.main_mw - r.baseline_mw).toFixed(2),
      },
    )
    return cols
  }, [compareKeysOnData])

  const compareColumns = useMemo(() => {
    const cols = [{ title: '时段', dataIndex: 'time', width: 72 }]
    const add = (prefix, dayArr, label) => {
      cols.push({ title: `${label}·主模型`, dataIndex: `${prefix}_main`, width: 112 })
      compareKeysOnData.forEach(({ key, short }) => {
        cols.push({ title: `${label}·${short}`, dataIndex: `${prefix}_${key}`, width: 100 })
      })
      cols.push({ title: `${label}·持久`, dataIndex: `${prefix}_base`, width: 92 })
    }
    add('d1', day1, '第1日')
    if (predSteps >= 192) add('d2', day2, '第2日')
    return cols
  }, [compareKeysOnData, predSteps, day1, day2])

  const compareData = useMemo(
    () =>
      Array.from({ length: 96 }, (_, i) => {
        const pad = (n) => String(n).padStart(2, '0')
        const h = Math.floor(i / 4)
        const m = (i % 4) * 15
        const row = {
          key: i,
          time: `${pad(h)}:${pad(m)}`,
        }
        const fill = (prefix, arr) => {
          const p = arr[i] || {}
          row[`${prefix}_main`] = p.main_mw
          row[`${prefix}_base`] = p.baseline_mw
          compareKeysOnData.forEach(({ key }) => {
            row[`${prefix}_${key}`] = p[key]
          })
        }
        fill('d1', day1)
        if (predSteps >= 192) fill('d2', day2)
        return row
      }),
    [day1, day2, predSteps, compareKeysOnData],
  )

  const chartRef = useRef(null)
  const [alertOpen, setAlertOpen] = useState(false)

  const exportPNG = () => {
    if (chartRef.current?.getEchartsInstance()) {
      const url = chartRef.current.getEchartsInstance().getDataURL({ type: 'png', pixelRatio: 2 })
      const a = document.createElement('a')
      a.href = url
      a.download = `负荷预测曲线_${startDate.format('YYYY-MM-DD')}.png`
      a.click()
      message.success('已导出 PNG')
    } else message.warning('请先加载图表')
  }

  const exportExcel = () => {
    const headers = ['时间', '本文主模型(MW)']
    compareKeysOnData.forEach(({ short }) => headers.push(`${short}(MW)`))
    headers.push('持久化(MW)', '主-持久(MW)')
    const rows = data.map((d) => {
      const r = [d.datetime, d.main_mw]
      compareKeysOnData.forEach(({ key }) => r.push(d[key] ?? ''))
      r.push(d.baseline_mw, (d.main_mw - d.baseline_mw).toFixed(2))
      return r
    })
    const csv = [headers.join(','), ...rows.map((r) => r.join(','))].join('\n')
    const blob = new Blob(['\ufeff' + csv], { type: 'text/csv;charset=utf-8' })
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = `负荷预测_${startDate.format('YYYY-MM-DD')}.csv`
    a.click()
    URL.revokeObjectURL(a.href)
    message.success('已导出 CSV')
  }

  useEffect(() => {
    if (data.length && data.some((d) => Math.abs(d.main_mw - d.baseline_mw) > 30)) {
      setAlertOpen(true)
    }
  }, [data])

  const compareInfoAlert =
    Array.isArray(compareModelsLoaded) && compareModelsLoaded.length === 0 ? (
      <Alert
        type="info"
        showIcon
        style={{ marginBottom: 16 }}
        message="当前仅展示主模型与持久化基线。若需在图中对比 LSTM / BiLSTM / TCN / BiLSTM-Attention，请在项目根目录运行完整训练流程以生成 预测模型/baseline_checkpoints/ 下权重，并重启后端。"
      />
    ) : null

  return (
    <div>
      <AlertModal
        open={alertOpen}
        onClose={() => setAlertOpen(false)}
        title="负荷偏差告警"
        content="检测到主模型与持久化基线预测偏差超过阈值，请关注。"
      />
      <div style={{ marginBottom: 24, display: 'flex', alignItems: 'center', gap: 16, flexWrap: 'wrap' }}>
        <span style={{ color: '#e8f4ff' }}>预测起始日期：</span>
        <DatePicker value={startDate} onChange={(d) => setStartDate(d || dayjs())} />
        <span style={{ color: '#e8f4ff' }}>预测长度：</span>
        <Select
          value={predSteps}
          onChange={setPredSteps}
          style={{ width: 220 }}
          options={[
            { value: 96, label: '快速（96 点 ≈ 1 天，推荐）' },
            { value: 192, label: '完整（192 点 = 2 天，较慢）' },
          ]}
        />
        <Button type="primary" className="tech-btn" onClick={loadData}>
          查询
        </Button>
        <Button onClick={exportPNG}>导出 PNG</Button>
        <Button onClick={exportExcel}>导出 Excel</Button>
        <Button type="default" onClick={() => (window.location.href = '/bigscreen')}>
          大屏模式
        </Button>
        {Array.isArray(compareModelsLoaded) && compareModelsLoaded.length > 0 && (
          <span style={{ color: 'rgba(180,220,255,0.9)', fontSize: 13 }}>
            已加载对比模型：{compareModelsLoaded.join('、')}
          </span>
        )}
        {weather && (
          <span style={{ color: 'rgba(232,244,255,0.85)', fontSize: 13, marginLeft: 8 }}>
            区域：{weather.location_label || weather.city_query || '—'}
            {weather.forecast?.[0] && (
              <>
                {' '}
                · 气象（
                {weather.source === 'openweather'
                  ? 'API'
                  : weather.source === 'training_climatology'
                    ? '气候学估计'
                    : weather.source === 'mean_fallback'
                      ? '均值兜底'
                      : weather.source || '—'}
                ）：
                {weather.forecast[0].description || '—'}，
                {weather.forecast[0].min_temp != null && weather.forecast[0].max_temp != null
                  ? `${Number(weather.forecast[0].min_temp).toFixed(1)}～${Number(weather.forecast[0].max_temp).toFixed(1)}°C`
                  : '—'}
                ，湿度{' '}
                {weather.forecast[0].humidity != null ? `${Number(weather.forecast[0].humidity).toFixed(0)}%` : '—'}
              </>
            )}
            {weather.error && <span style={{ color: '#faad14' }}>（{weather.error}）</span>}
          </span>
        )}
      </div>
      {compareInfoAlert}
      {loadHint && <Alert type="warning" showIcon message={loadHint} style={{ marginBottom: 16 }} />}
      <Spin spinning={loading}>
        <Tabs
          items={[
            {
              key: 'chart',
              label: '折线图',
              children: data.length ? (
                <ReactECharts
                  key={(data[0]?.datetime || 'curve') + String(compareKeysOnData.length)}
                  ref={chartRef}
                  option={chartOption}
                  style={{ height: 440 }}
                  notMerge
                  lazyUpdate
                />
              ) : (
                <Empty description={loading ? '加载中…' : '暂无预测曲线'} style={{ padding: 48 }} />
              ),
            },
            {
              key: 'table',
              label: '预测表格',
              children: (
                <Table
                  columns={tableColumns}
                  dataSource={data}
                  pagination={{ pageSize: 24 }}
                  size="small"
                  scroll={{ x: 720 + compareKeysOnData.length * 112 }}
                />
              ),
            },
            {
              key: 'compare',
              label: '两日对比',
              children: (
                <div>
                  {predSteps < 192 && (
                    <Alert
                      type="info"
                      showIcon
                      style={{ marginBottom: 12 }}
                      message="当前为快速预览（少于 192 点），「第二日」列无数据。需要两日对比请选择「完整（192 点）」后重新查询。"
                    />
                  )}
                  <Table
                    columns={compareColumns}
                    dataSource={compareData}
                    pagination={{ pageSize: 24 }}
                    size="small"
                    scroll={{ x: Math.max(900, compareColumns.length * 98) }}
                  />
                </div>
              ),
            },
          ]}
        />
      </Spin>
    </div>
  )
}
