import { useState, useEffect } from 'react'
import { Card, List, message } from 'antd'
import { messagesApi } from '../api'

export default function MemberMessages() {
  const [messages, setMessages] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    messagesApi.list().then((res) => {
      setMessages(res.data?.data || [])
    }).catch(() => message.error('加载失败')).finally(() => setLoading(false))
  }, [])

  return (
    <Card title="收到的消息">
      <List
        loading={loading}
        dataSource={messages}
        renderItem={(m) => (
          <List.Item>
            <List.Item.Meta
              title={`来自 ${m.sender_name || '系统'} · ${m.created_at ? new Date(m.created_at).toLocaleString() : ''}`}
              description={m.content}
            />
          </List.Item>
        )}
      />
    </Card>
  )
}
