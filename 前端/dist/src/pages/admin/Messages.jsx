import { useState, useEffect } from 'react'
import { Card, Input, Button, Select, Radio, message } from 'antd'
import { adminApi } from '../../api'

const { TextArea } = Input

export default function AdminMessages() {
  const [content, setContent] = useState('')
  const [mode, setMode] = useState('all')
  const [selectedIds, setSelectedIds] = useState([])
  const [users, setUsers] = useState([])
  const [loading, setLoading] = useState(false)
  const [sendLoading, setSendLoading] = useState(false)

  const user = JSON.parse(localStorage.getItem('user') || '{}')
  const isTotalAdmin = user.role === 'total_admin'

  useEffect(() => {
    adminApi.listUsers().then((res) => {
      const list = Array.isArray(res.data) ? res.data : (res.data?.data || [])
      setUsers(list.filter((u) => !u.disabled))
    }).catch(() => message.error('加载用户列表失败'))
  }, [])

  const send = async () => {
    if (!content.trim()) {
      message.warning('请输入消息内容')
      return
    }
    const receiverIds = mode === 'all' ? ['all'] : selectedIds
    if (mode === 'select' && receiverIds.length === 0) {
      message.warning('请选择接收人')
      return
    }
    setSendLoading(true)
    try {
      await adminApi.sendMessage({ receiver_ids: receiverIds, content: content.trim() })
      message.success(mode === 'all' ? '已发送给所有人' : `已发送给 ${receiverIds.length} 人`)
      setContent('')
      setSelectedIds([])
    } catch (e) {
      message.error(e.response?.data?.detail || '发送失败')
    } finally {
      setSendLoading(false)
    }
  }

  const userOptions = users.map((u) => ({ value: u.id, label: `${u.username} (${u.role === 'group_admin' ? '小组管理员' : u.role === 'total_admin' ? '总管理员' : '成员'})` }))
  const groupMembers = isTotalAdmin ? users : users.filter((u) => u.group_id === user.group_id)

  return (
    <Card className="glass-card" title={isTotalAdmin ? '发送消息（可发全体或指定用户）' : '发送消息（可发全体组员或指定组员）'}>
      <div style={{ marginBottom: 16 }}>
        <span style={{ color: '#e8f4ff', marginRight: 8 }}>接收范围：</span>
        <Radio.Group value={mode} onChange={(e) => setMode(e.target.value)} style={{ marginBottom: 8 }}>
          <Radio value="all">{isTotalAdmin ? '全体用户' : '全体组员'}</Radio>
          <Radio value="select">选择接收人</Radio>
        </Radio.Group>
      </div>
      {mode === 'select' && (
        <div style={{ marginBottom: 16 }}>
          <span style={{ color: '#e8f4ff', marginRight: 8 }}>选择接收人：</span>
          <Select
            mode="multiple"
            placeholder={isTotalAdmin ? '选择用户' : '选择组员'}
            options={isTotalAdmin ? userOptions : groupMembers.map((u) => ({ value: u.id, label: u.username }))}
            value={selectedIds}
            onChange={setSelectedIds}
            style={{ width: 400 }}
          />
        </div>
      )}
      <div style={{ marginBottom: 16 }}>
        <span style={{ color: '#e8f4ff', marginRight: 8 }}>消息内容：</span>
        <TextArea rows={4} value={content} onChange={(e) => setContent(e.target.value)} placeholder="输入消息内容" />
      </div>
      <Button type="primary" className="tech-btn" onClick={send} loading={sendLoading}>发送</Button>
    </Card>
  )
}
