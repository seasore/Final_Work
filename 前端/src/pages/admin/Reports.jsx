import { useState, useEffect } from 'react'
import { Tabs, Table, Button, Modal, Input, Select, message } from 'antd'
import { adminApi } from '../../api'

const URGENCY_MAP = { '1': '低', '2': '中', '3': '高', '4': '紧急' }

export default function AdminReports() {
  const [unread, setUnread] = useState([])
  const [read, setRead] = useState([])
  const [loading, setLoading] = useState(true)
  const [urgencyFilter, setUrgencyFilter] = useState('')
  const [replyModal, setReplyModal] = useState(null)
  const [replyContent, setReplyContent] = useState('')

  const load = async () => {
    setLoading(true)
    try {
      const [u, r] = await Promise.all([
        adminApi.unreadReports(urgencyFilter || undefined),
        adminApi.readReports(urgencyFilter || undefined),
      ])
      setUnread(u.data?.data || [])
      setRead(r.data?.data || [])
    } catch (e) {
      message.error('加载失败')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => load(), [urgencyFilter])

  const submitReply = async () => {
    if (!replyModal || !replyContent.trim()) return
    try {
      await adminApi.replyReport(replyModal.id, replyContent)
      message.success('已回复')
      setReplyModal(null)
      setReplyContent('')
      load()
    } catch (e) {
      message.error(e.response?.data?.detail || '回复失败')
    }
  }

  const columns = [
    { title: '汇报人', dataIndex: 'reporter_name', width: 100 },
    { title: '紧急程度', dataIndex: 'urgency', width: 80, render: (u) => URGENCY_MAP[u] || u },
    { title: '内容', dataIndex: 'content', ellipsis: true },
    { title: '时间', dataIndex: 'created_at', width: 180, render: (t) => t ? new Date(t).toLocaleString() : '' },
    {
      title: '操作',
      width: 80,
      render: (_, r) => !r.replied && (
        <Button size="small" type="link" onClick={() => setReplyModal(r)}>回复</Button>
      ),
    },
  ]

  const readColumns = [
    ...columns.slice(0, -1),
    { title: '回复', dataIndex: 'reply_content', ellipsis: true },
  ]

  return (
    <div>
      <div style={{ marginBottom: 16 }}>
        <span>按紧急程度筛选：</span>
        <Select value={urgencyFilter} onChange={setUrgencyFilter} style={{ width: 120 }} allowClear>
          <Select.Option value="">全部</Select.Option>
          {Object.entries(URGENCY_MAP).map(([k, v]) => <Select.Option key={k} value={k}>{v}</Select.Option>)}
        </Select>
      </div>
      <Tabs
        items={[
          { key: 'unread', label: `未读汇报 (${unread.length})`, children: <Table columns={columns} dataSource={unread} rowKey="id" loading={loading} /> },
          { key: 'read', label: '已读汇报', children: <Table columns={readColumns} dataSource={read} rowKey="id" loading={loading} /> },
        ]}
      />
      <Modal
        title="回复汇报"
        open={!!replyModal}
        onOk={submitReply}
        onCancel={() => setReplyModal(null)}
      >
        {replyModal && <p>{replyModal.content}</p>}
        <Input.TextArea rows={4} value={replyContent} onChange={(e) => setReplyContent(e.target.value)} placeholder="回复内容" />
      </Modal>
    </div>
  )
}
