import { useState, useEffect } from 'react'
import { Table, message } from 'antd'
import { adminApi } from '../../api'

export default function AdminLogs() {
  const [logs, setLogs] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    adminApi.logs(200).then((res) => {
      setLogs(res.data?.data || [])
    }).catch(() => message.error('加载失败')).finally(() => setLoading(false))
  }, [])

  const columns = [
    { title: '用户ID', dataIndex: 'user_id', width: 120 },
    { title: '操作', dataIndex: 'action', width: 120 },
    { title: '详情', dataIndex: 'detail' },
    { title: '时间', dataIndex: 'created_at', width: 180, render: (t) => t ? new Date(t).toLocaleString() : '' },
  ]

  return <Table columns={columns} dataSource={logs} rowKey="id" loading={loading} />
}
