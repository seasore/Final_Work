import { useEffect, useState } from 'react'
import { Card, Table, Button, DatePicker, Spin, message } from 'antd'
import dayjs from 'dayjs'
import { analyticsApi } from '../api'

export default function Assessment() {
  const [month, setMonth] = useState(dayjs())

  const [loading, setLoading] = useState(false)
  const [rows, setRows] = useState([])

  const load = async () => {
    setLoading(true)
    try {
      const y = month.year()
      const m = month.month() + 1
      const res = await analyticsApi.assessment(y, m)
      setRows(res.data.rows || [])
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
        <span style={{ color: '#e8f4ff', marginRight: 8 }}>月份：</span>
        <DatePicker picker="month" value={month} onChange={(v) => setMonth(v || dayjs())} style={{ marginRight: 16 }} />
        <Button type="primary" className="tech-btn" loading={loading} onClick={load}>生成考核报表</Button>
    </Card>
      <Spin spinning={loading}>
        <Card className="glass-card" title="考核报表（基于训练集抽样回测）">
          <Table
            dataSource={rows}
            columns={[
              { title: '区域', dataIndex: 'unit' },
              { title: '月份', dataIndex: 'month' },
              { title: '平均准确率', dataIndex: 'accuracy' },
              { title: '排名', dataIndex: 'rank' },
              { title: '样本点数', dataIndex: 'sample_points' },
            ]}
          />
        </Card>
      </Spin>
    </div>
  )
}
