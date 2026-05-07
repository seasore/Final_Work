import { useState, useEffect } from 'react'
import { Card, Form, Input, Select, Button, Table, message } from 'antd'
import { reportsApi } from '../api'

const URGENCY_OPTIONS = [
  { value: '1', label: '低' },
  { value: '2', label: '中' },
  { value: '3', label: '高' },
  { value: '4', label: '紧急' },
]

export default function MemberReports() {
  const [reports, setReports] = useState([])
  const [loading, setLoading] = useState(true)
  const [form] = Form.useForm()

  const load = () => {
    setLoading(true)
    reportsApi.myReports().then((res) => {
      setReports(res.data?.data || [])
    }).catch(() => message.error('加载失败')).finally(() => setLoading(false))
  }

  useEffect(() => { load() }, [])

  const submit = async () => {
    try {
      const v = await form.validateFields()
      await reportsApi.create(v)
      message.success('汇报已提交')
      form.resetFields()
      load()
    } catch (e) {
      if (e.errorFields) return
      message.error(e.response?.data?.detail || '提交失败')
    }
  }

  return (
    <div>
      <Card title="向上汇报" style={{ marginBottom: 24 }}>
        <Form form={form} layout="inline" onFinish={submit}>
          <Form.Item name="content" rules={[{ required: true }]} style={{ flex: 1, minWidth: 200 }}>
            <Input.TextArea placeholder="汇报内容" rows={2} style={{ minWidth: 300 }} />
          </Form.Item>
          <Form.Item name="urgency" label="紧急程度" rules={[{ required: true }]}>
            <Select options={URGENCY_OPTIONS} style={{ width: 100 }} />
          </Form.Item>
          <Form.Item>
            <Button type="primary" htmlType="submit">提交</Button>
          </Form.Item>
        </Form>
      </Card>
      <Card title="我的汇报记录">
        <Table
          dataSource={reports}
          rowKey="id"
          loading={loading}
          columns={[
            { title: '紧急程度', dataIndex: 'urgency', width: 80, render: (u) => URGENCY_OPTIONS.find(o => o.value === u)?.label || u },
            { title: '内容', dataIndex: 'content' },
            { title: '回复', dataIndex: 'reply_content' },
            { title: '时间', dataIndex: 'created_at', width: 180, render: (t) => t ? new Date(t).toLocaleString() : '' },
          ]}
        />
      </Card>
    </div>
  )
}
