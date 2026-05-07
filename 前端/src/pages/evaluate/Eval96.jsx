import { useRef, useState } from 'react'
import { Card, DatePicker, Button, Row, Col, Spin, message } from 'antd'
import ReactECharts from 'echarts-for-react'
import dayjs from 'dayjs'
import { analyticsApi } from '../../api'

const HEATMAP_GRADIENT = ['#1890ff', '#40a9ff', '#69c0ff', '#91d5ff', '#bae7ff', '#fff3cd', '#ffe0b2', '#ffcc80', '#ffb74d', '#ff9800']

export default function Eval96() {
  const heatmapRef = useRef(null)
  const [day, setDay] = useState(dayjs('2024-03-01'))
  const [loading, setLoading] = useState(false)
  const [cells, setCells] = useState([])
  const [meanMain, setMeanMain] = useState(null)
  const [meanBase, setMeanBase] = useState(null)
  const [label, setLabel] = useState('')

  const load = async () => {
    setLoading(true)
    try {
      const res = await analyticsApi.heatmap96(day.format('YYYY-MM-DD'))
      setCells(res.data.cells || [])
      setMeanMain(res.data.mean_accuracy_main)
      setMeanBase(res.data.mean_accuracy_baseline)
      setLabel(res.data.location_label || '')
      message.success('已加载单日 96 点回测')
    } catch (e) {
      message.error(e.response?.data?.detail || '加载失败')
      setCells([])
    } finally {
      setLoading(false)
    }
  }

  const vals = cells.map((c) => c[2])
  let vmin = vals.length ? Math.min(...vals) : 0
  let vmax = vals.length ? Math.max(...vals) : 100
  if (vmin === vmax) vmax = vmin + 1e-6

  const heatmapOption = {
    backgroundColor: 'transparent',
    title: { text: `96 点准确率热力图 · ${label}`, textStyle: { color: '#e8f4ff' } },
    tooltip: {},
    xAxis: { type: 'category', data: ['0-15分', '15-30分', '30-45分', '45-60分'], axisLabel: { color: '#e8f4ff' } },
    yAxis: { type: 'category', data: Array.from({ length: 24 }, (_, i) => `${i}时`), axisLabel: { color: '#e8f4ff' } },
    visualMap: { min: vmin, max: vmax, inRange: { color: HEATMAP_GRADIENT }, textStyle: { color: '#e8f4ff' } },
    series: [{ type: 'heatmap', data: cells }],
  }

  const barOption = {
    backgroundColor: 'transparent',
    title: { text: '单日平均准确率', textStyle: { color: '#e8f4ff' } },
    xAxis: { type: 'category', data: ['TCN-BiLSTM-Attention', '持久化模型'], axisLabel: { color: '#e8f4ff' } },
    yAxis: { type: 'value', min: 0, max: 100, axisLabel: { color: '#e8f4ff' } },
    series: [
      {
        type: 'bar',
        data: [meanMain ?? 0, meanBase ?? 0],
        itemStyle: { color: (params) => (params.dataIndex === 0 ? '#40a9ff' : '#ff4d4f') },
      },
    ],
  }

  const exportPNG = () => {
    if (heatmapRef.current?.getEchartsInstance()) {
      const url = heatmapRef.current.getEchartsInstance().getDataURL({ type: 'png', pixelRatio: 2 })
      const a = document.createElement('a')
      a.href = url
      a.download = '96点准确率热力图.png'
      a.click()
      message.success('已导出 PNG')
    }
  }

  return (
    <div>
      <Card className="glass-card" style={{ marginBottom: 24 }}>
        <Row gutter={16} align="middle">
          <Col><span style={{ color: '#e8f4ff' }}>回测日期：</span><DatePicker value={day} onChange={(d) => setDay(d || dayjs())} /></Col>
          <Col><Button type="primary" className="tech-btn" loading={loading} onClick={load}>查询</Button></Col>
          <Col><Button onClick={exportPNG}>导出 PNG</Button></Col>
        </Row>
        <div style={{ color: 'rgba(255,255,255,0.55)', fontSize: 12, marginTop: 8 }}>需选择训练集 2023-01-01～2024-04-22 内且后续有足够样本的日期。</div>
      </Card>
      <Spin spinning={loading}>
        <Row gutter={24}>
          <Col span={14}><Card className="glass-card"><ReactECharts ref={heatmapRef} option={heatmapOption} style={{ height: 400 }} notMerge /></Card></Col>
          <Col span={10}><Card className="glass-card"><ReactECharts option={barOption} style={{ height: 400 }} notMerge /></Card></Col>
        </Row>
      </Spin>
    </div>
  )
}
