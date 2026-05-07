import { useEffect, useState } from 'react'
import { Card, Select, Button, Spin, message } from 'antd'
import ReactECharts from 'echarts-for-react'
import { analyticsApi } from '../../api'

export default function EvalHoliday() {
  const [year, setYear] = useState(2024)
  const [loading, setLoading] = useState(true)
  const [labels, setLabels] = useState([])
  const [accMain, setAccMain] = useState([])
  const [accBase, setAccBase] = useState([])
  const [loc, setLoc] = useState('')

  const load = async () => {
    setLoading(true)
    try {
      const [l, acc] = await Promise.all([analyticsApi.location(), analyticsApi.holidayAccuracy(year)])
      setLoc(l.data?.label || '')
      setLabels(acc.data.labels || [])
      setAccMain(acc.data.accuracy_main || [])
      setAccBase(acc.data.accuracy_baseline || [])
    } catch (e) {
      message.error(e.response?.data?.detail || '加载失败')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { load() }, [year])

  const chartOption = {
    backgroundColor: 'transparent',
    title: { text: `节假日回测准确率（%）· ${loc}`, textStyle: { color: '#e8f4ff' } },
    xAxis: { type: 'category', data: labels, axisLabel: { color: '#e8f4ff', rotate: 25 } },
    yAxis: { type: 'value', min: 0, max: 100, axisLabel: { color: '#e8f4ff' } },
    series: [
      { type: 'bar', data: accMain, name: 'TCN-BiLSTM', itemStyle: { color: '#40a9ff' } },
      { type: 'bar', data: accBase, name: '持久化模型', itemStyle: { color: '#ff4d4f' } },
    ],
    legend: { textStyle: { color: '#e8f4ff' } },
  }

  return (
    <div>
      <Card className="glass-card" style={{ marginBottom: 24 }}>
        <span style={{ color: '#e8f4ff', marginRight: 8 }}>年份：</span>
        <Select style={{ width: 100, marginRight: 16 }} value={year} onChange={setYear} options={[2024, 2023, 2022].map((y) => ({ value: y, label: String(y) }))} />
        <Button type="primary" className="tech-btn" onClick={load}>刷新</Button>
      </Card>
      <Spin spinning={loading}>
        <Card className="glass-card"><ReactECharts option={chartOption} style={{ height: 350 }} notMerge /></Card>
      </Spin>
    </div>
  )
}
