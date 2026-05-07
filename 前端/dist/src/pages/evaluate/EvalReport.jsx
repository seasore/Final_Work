import { useState } from 'react'
import { Card, Select, DatePicker, Button, Table, Spin, message } from 'antd'
import dayjs from 'dayjs'
import { analyticsApi } from '../../api'

export default function EvalReport() {
  const [dateRange, setDateRange] = useState([dayjs().subtract(1, 'day'), dayjs().subtract(1, 'day')])
  const [mode, setMode] = useState('moment')
  const [loading, setLoading] = useState(false)
  const [rows, setRows] = useState([])
  const [note, setNote] = useState('')

  const handleQuery = async () => {
    const start = dateRange?.[0]?.format('YYYY-MM-DD')
    const end = dateRange?.[1]?.format('YYYY-MM-DD') || start
    if (!start) {
      message.warning('请选择日期')
      return
    }
    const days = dayjs(end).diff(dayjs(start), 'day') + 1
    if (days > 3) {
      message.warning('为控制计算时间，单次最多回测 3 天，请缩小范围')
      return
    }
    setLoading(true)
    try {
      const res = await analyticsApi.backtest(start, end)
      setNote(res.data.note || '')
      const list = (res.data.rows || []).map((r, i) => {
        const [d, t] = r.datetime.split(' ')
        return {
          key: i,
          date: d,
          time: t,
          main: r.main_mw,
          actual: r.actual,
          diff: (r.main_mw - r.actual).toFixed(2),
          accuracy: `${r.accuracy_main_pct}%`,
        }
      })
      setRows(list)
      message.success(`已加载 ${list.length} 条回测点`)
    } catch (e) {
      message.error(e.response?.data?.detail || '回测失败')
      setRows([])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div>
      <Card className="glass-card" style={{ marginBottom: 24 }}>
        <span style={{ color: '#e8f4ff', marginRight: 8 }}>切换：</span>
        <Select style={{ width: 120, marginRight: 16 }} value={mode} onChange={setMode} options={[{ value: 'moment', label: '时刻准确率' }, { value: 'day', label: '多日' }, { value: 'month', label: '多月' }, { value: 'year', label: '多年' }]} />
        <DatePicker.RangePicker value={dateRange} onChange={(v) => setDateRange(v || [dayjs(), dayjs()])} style={{ marginRight: 16 }} />
        <Button type="primary" className="tech-btn" loading={loading} onClick={handleQuery}>查询回测</Button>
        {note && <span style={{ color: 'rgba(255,255,255,0.55)', marginLeft: 16, fontSize: 12 }}>{note}</span>}
      </Card>
      <Card className="glass-card" title="历史回测（训练集实际负荷 vs 模型预测）">
        <Spin spinning={loading}>
          <Table
            dataSource={rows}
            columns={[
              { title: '日期', dataIndex: 'date', width: 110 },
              { title: '时刻', dataIndex: 'time', width: 80 },
              { title: '主模型预测(MW)', dataIndex: 'main' },
              { title: '实际负荷(MW)', dataIndex: 'actual' },
              { title: '偏差(MW)', dataIndex: 'diff' },
              { title: '准确率(主)', dataIndex: 'accuracy' },
            ]}
            pagination={{ pageSize: 24 }}
            size="small"
          />
        </Spin>
      </Card>
    </div>
  )
}
