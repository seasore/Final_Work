import { useEffect, useState } from 'react'
import { Card, Table, Button, DatePicker, Spin, message } from 'antd'
import dayjs from 'dayjs'
import { reportsApi, analyticsApi } from '../../api'

const URGENCY = { '1': '低', '2': '中', '3': '高', '4': '紧急' }

export default function ReportPredict() {
  const [range, setRange] = useState([dayjs().subtract(30, 'day'), dayjs()])
  const [loading, setLoading] = useState(true)
  const [rows, setRows] = useState([])
  const [locationLabel, setLocationLabel] = useState('')

  const load = async () => {
    setLoading(true)
    try {
      const [loc, rep] = await Promise.all([
        analyticsApi.location(),
        reportsApi.myReports(),
      ])
      setLocationLabel(loc.data?.label || '')
      const list = (rep.data.data || []).map((r, i) => {
        const t = r.created_at ? dayjs(r.created_at) : null
        return {
          key: r.id || i,
          date: t ? t.format('YYYY-MM-DD HH:mm') : '—',
          unit: loc.data?.label || '—',
          urgency: URGENCY[r.urgency] || r.urgency,
          status: r.replied ? '已回复' : '待回复',
          preview: (r.content || '').slice(0, 40),
        }
      })
      const [a, b] = range
      const filtered = list.filter((x) => {
        if (!a || !b) return true
        const d = dayjs(x.date, 'YYYY-MM-DD HH:mm')
        if (!d.isValid()) return true
        const t0 = a.startOf('day').valueOf()
        const t1 = b.endOf('day').valueOf()
        const t = d.valueOf()
        return t >= t0 && t <= t1
      })
      setRows(filtered)
    } catch (e) {
      message.error(e.response?.data?.detail || '加载失败')
      setRows([])
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { load() }, [])

  return (
    <div>
      <Card className="glass-card" style={{ marginBottom: 24 }}>
        <span style={{ color: '#e8f4ff', marginRight: 8 }}>区域：</span><span style={{ color: '#91d5ff', marginRight: 16 }}>{locationLabel || '—'}</span>
        <span style={{ color: '#e8f4ff', marginRight: 8 }}>日期：</span>
        <DatePicker.RangePicker value={range} onChange={setRange} style={{ marginRight: 16 }} />
        <Button type="primary" className="tech-btn" onClick={load}>查询</Button>
      </Card>
      <Spin spinning={loading}>
        <Card className="glass-card" title="我的汇报（MongoDB 真实数据）">
          <Table
            dataSource={rows}
            columns={[
              { title: '时间', dataIndex: 'date', width: 160 },
              { title: '区域', dataIndex: 'unit' },
              { title: '紧急程度', dataIndex: 'urgency', width: 90 },
              { title: '状态', dataIndex: 'status', width: 90 },
              { title: '内容摘要', dataIndex: 'preview' },
            ]}
          />
        </Card>
      </Spin>
    </div>
  )
}
